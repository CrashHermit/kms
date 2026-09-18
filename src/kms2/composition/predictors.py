"""Factories for configured and optionally recorded DSPy predictors."""

import dspy

from kms2.config.inference import StageInferenceSettings
from kms2.local_models import LocalModelRuntime
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
        inference: StageInferenceSettings,
        signature: type[dspy.Signature],
    ) -> dspy.Module:
        """Create one configured predictor, optionally wrapped for recording."""
        predictor = self._local_models.predictor(inference, signature)
        if self._recorder is None:
            return predictor
        return RecordingModule(predictor, signature, self._recorder)
