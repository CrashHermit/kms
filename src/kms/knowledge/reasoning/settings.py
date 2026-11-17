"""Inference and evidence limits for proposition reasoning."""

from pydantic import BaseModel, ConfigDict, Field

from kms.config.inference import NoRetryTextInferenceSettings


class ReasoningSettings(BaseModel):
    """Configure text inference, input capacity, and candidate limits."""

    model_config = ConfigDict(extra='forbid')

    inference: NoRetryTextInferenceSettings = Field(
        default_factory=NoRetryTextInferenceSettings
    )
    input_token_budget: int = Field(default=8192, gt=0)
    safety_margin_tokens: int = Field(default=256, ge=0)
    candidate_limit: int = Field(default=64, gt=0)
    argument_limit: int = Field(default=64, gt=0)
