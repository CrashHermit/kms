"""Factories for configured and optionally recorded DSPy predictors."""

import asyncio
from collections.abc import Sequence

import dspy

from kms2.config.inference import PredictorStrategy, StageInferenceSettings
from kms2.config.llm import ChatGPTLLMProfile, LocalLLMProfile
from kms2.config.settings import Settings
from kms2.core.windowing import TokenBudget
from kms2.inference.token_counting import InferenceTokenizers
from kms2.local_models.runtime import LocalModelRuntime
from kms2.train.recorder import Recorder, RecordingModule


class _BoundedCloudPredictor(dspy.Module):
    """Bound subscription inference across modules from one factory."""

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
        """Bound concurrent cloud inference through the factory semaphore."""
        async with self._semaphore:
            return await self.predictor.acall(**kwargs)


class PredictorFactory:
    """Create runtime predictors with optional training-example recording."""

    def __init__(
        self,
        settings: Settings,
        local_models: LocalModelRuntime,
        recorder: Recorder | None,
    ) -> None:
        self._settings = settings
        self._local_models = local_models
        self._recorder = recorder
        self._cloud_semaphore = asyncio.Semaphore(settings.llm.max_concurrency)

    def create(
        self,
        module: type[dspy.Module],
        inference: StageInferenceSettings,
        signature: type[dspy.Signature],
        *,
        examples: Sequence[dspy.Example] | None = None,
    ) -> dspy.Module:
        """Attach explicit runtime examples before concurrency and recording wrappers."""
        profile = self._settings.llm.resolve(inference)
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
            predictor = _build_strategy(inference.strategy, signature)
            if examples is not None:
                for parameter in predictor.predictors():
                    parameter.demos = list(examples)
            predictor.set_lm(language_model)
            configured: dspy.Module = _BoundedCloudPredictor(
                predictor,
                self._cloud_semaphore,
            )
        else:
            assert isinstance(profile, LocalLLMProfile)
            configured = self._local_models.predictor(
                inference,
                signature,
                model_server_profile=profile.model_server_profile,
                examples=examples,
            )
        if self._recorder is None:
            return configured
        return RecordingModule(configured, module, signature, self._recorder)


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
    tokenizers: InferenceTokenizers,
    inference: StageInferenceSettings,
    *,
    input_token_budget: int,
    safety_margin_tokens: int,
) -> TokenBudget:
    """Build a profile-aware application-payload budget for one stage."""
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
    return TokenBudget(
        counter=tokenizers.for_inference(inference),
        token_limit=min(input_token_budget, token_limit),
    )
