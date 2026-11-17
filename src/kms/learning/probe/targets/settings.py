"""Inference settings for atomic target decomposition."""

from pydantic import BaseModel, ConfigDict, Field

from kms.config.inference import TextInferenceSettings


class TargetDecompositionSettings(BaseModel):
    """Text inference for decomposing a clarified learning goal."""

    model_config = ConfigDict(extra='forbid')

    inference: TextInferenceSettings = Field(
        default_factory=lambda: TextInferenceSettings(
            max_tokens=2048, num_retries=0
        )
    )
