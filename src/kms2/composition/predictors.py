"""Factories for configured and optionally recorded DSPy predictors."""

import dspy

from kms2.config.inference import StageInferenceSettings
from kms2.config.runtime import LocalModelRuntimeSettings
from kms2.core.windowing import TokenBudget
from kms2.local_models.runtime import LocalModelRuntime
from kms2.local_models.token_counting import LocalTokenizers
from kms2.train.recorder import Recorder, RecordingModule


class PredictorFactory:
    """Create runtime predictors with optional training-example recording."""

    def __init__(
        self,
        local_models: LocalModelRuntime,
        recorder: Recorder | None,
    ) -> None:
        self._local_models = local_models
        self._recorder = recorder

    def create(
        self,
        module: type[dspy.Module],
        inference: StageInferenceSettings,
        signature: type[dspy.Signature],
    ) -> dspy.Module:
        """Create one configured predictor, optionally wrapped for recording."""
        predictor = self._local_models.predictor(inference, signature)
        if self._recorder is None:
            return predictor
        return RecordingModule(predictor, module, signature, self._recorder)


def build_token_budget(
    settings: LocalModelRuntimeSettings,
    tokenizers: LocalTokenizers,
    inference: StageInferenceSettings,
    *,
    input_token_budget: int,
    safety_margin_tokens: int,
) -> TokenBudget:
    """Build an estimated application-payload budget for one stage."""
    profile = settings.router.model_server_profiles[
        inference.model_server_profile
    ]
    capacity = profile.context_size // profile.parallel
    token_limit = min(
        input_token_budget,
        capacity - inference.max_tokens - safety_margin_tokens,
    )
    return TokenBudget(
        counter=tokenizers.for_profile(inference.model_server_profile),
        token_limit=token_limit,
    )
