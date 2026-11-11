"""Factories for configured and optionally recorded DSPy predictors."""

import asyncio
from collections.abc import Sequence

import dspy

from kms.config.inference import PredictorStrategy, StageInferenceSettings
from kms.config.llm import ChatGPTLLMProfile, LocalLLMProfile
from kms.config.settings import Settings
from kms.core.tokenization.budget import TokenBudget
from kms.inference.generation import (
    NativePredictor,
    generation_prompt,
    generation_signature,
    prepare_generation_examples,
)
from kms.inference.tokenization.resources import LanguageModelTokenizers
from kms.local_models.runtime import LocalModelRuntime
from kms.media.budget import ImagePayloadBudget
from kms.train.recorder import Recorder, RecordingModule


class _BoundedPredictor(dspy.Module):
    """Bound preparation and execution across one invocation's generation tasks."""

    def __init__(
        self,
        predictor: dspy.Module,
        semaphore: asyncio.Semaphore,
    ) -> None:
        super().__init__()
        self.predictor = predictor
        self._semaphore = semaphore

    def forward(self, **kwargs: object) -> object:
        """Run synchronous DSPy work outside an active event loop."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.acall(**kwargs))
        raise RuntimeError('use acall() from an active event loop')

    async def aforward(self, **kwargs: object) -> object:
        """Hold capacity while preparing media and awaiting the backend."""
        async with self._semaphore:
            return await self.predictor.acall(**kwargs)


class PredictorFactory:
    """Create runtime predictors with optional training-example recording."""

    def __init__(
        self,
        settings: Settings,
        local_models: LocalModelRuntime,
        recorder: Recorder | None,
        *,
        tokenizers: LanguageModelTokenizers,
    ) -> None:
        self._settings = settings
        self._local_models = local_models
        self._recorder = recorder
        self._semaphore = asyncio.Semaphore(settings.llm.max_concurrency)
        self._tokenizers = tokenizers

    def create(
        self,
        module: type[dspy.Module],
        inference: StageInferenceSettings,
        signature: type[dspy.Signature],
        *,
        examples: Sequence[dspy.Example] | None = None,
        budget: TokenBudget | None = None,
        image_budget: ImagePayloadBudget | None = None,
    ) -> dspy.Module:
        """Compose prompt-aware capacity, native preparation and backend execution."""
        profile = self._settings.llm.resolve(inference)
        if budget is None:
            budget = build_token_budget(
                self._settings,
                self._tokenizers,
                inference,
                safety_margin_tokens=0,
                signature=signature,
                examples=examples,
            )
        native_examples = (
            prepare_generation_examples(examples, signature)
            if examples is not None
            else None
        )
        native_signature = generation_signature(signature)
        if isinstance(profile, ChatGPTLLMProfile):
            language_model = dspy.LM(
                f'chatgpt/{profile.model}',
                model_type='responses',
                temperature=None,
                max_tokens=None,
                reasoning_effort=profile.reasoning_effort,
                num_retries=inference.num_retries,
                cache=True,
                engine='litellm',
                use_developer_role=True,
            )
            predictor = _build_strategy(inference.strategy, native_signature)
            if native_examples is not None:
                for parameter in predictor.predictors():
                    parameter.demos = native_examples
            predictor.set_lm(language_model)
            configured: dspy.Module = predictor
        else:
            assert isinstance(profile, LocalLLMProfile)
            configured = self._local_models.predictor(
                inference,
                native_signature,
                model_server_profile=profile.model_server_profile,
                examples=native_examples,
            )
        prepared: dspy.Module = NativePredictor(
            configured,
            native_signature,
            budget=budget,
            image_budget=image_budget,
            supports_images=(
                isinstance(profile, ChatGPTLLMProfile)
                or self._settings.local_models.router.model_server_profiles[
                    profile.model_server_profile
                ].mmproj_path
                is not None
            ),
        )
        if self._recorder is not None:
            prepared = RecordingModule(
                prepared, module, signature, self._recorder
            )
        return _BoundedPredictor(prepared, self._semaphore)


def _build_strategy(
    strategy: PredictorStrategy,
    signature: type[dspy.Signature],
) -> dspy.Module:
    """Build the selected DSPy inference strategy."""
    if strategy is PredictorStrategy.PREDICT:
        return dspy.Predict(signature)
    return dspy.ChainOfThought(signature)


def build_token_budget(
    settings: Settings,
    tokenizers: LanguageModelTokenizers,
    inference: StageInferenceSettings,
    *,
    input_token_budget: int | None = None,
    safety_margin_tokens: int,
    signature: type[dspy.Signature] | None = None,
    examples: Sequence[dspy.Example] | None = None,
) -> TokenBudget:
    """Reserve model output capacity separately from the live workload limit."""
    profile = settings.llm.resolve(inference)
    if isinstance(profile, LocalLLMProfile):
        local_profile = settings.local_models.router.model_server_profiles[
            profile.model_server_profile
        ]
        token_limit = (
            local_profile.context_size // local_profile.parallel
            - inference.max_tokens
            - safety_margin_tokens
        )
    else:
        assert isinstance(profile, ChatGPTLLMProfile)
        token_limit = min(
            profile.max_input_tokens,
            profile.context_size - inference.max_tokens - safety_margin_tokens,
        )
    prompt = None
    prompt_images = ()
    if signature is not None:
        prompt_signature = signature
        if inference.strategy is PredictorStrategy.CHAIN_OF_THOUGHT:
            prompt_signature = dspy.ChainOfThought(signature).predict.signature
        prompt, prompt_images = generation_prompt(
            prompt_signature, examples or ()
        )
    return TokenBudget(
        counter=tokenizers.for_inference(inference),
        token_limit=(
            input_token_budget
            if input_token_budget is not None
            else token_limit
        ),
        format_request=prompt,
        context_token_limit=token_limit,
        prompt_images=prompt_images,
    )
