"""Inference and context budgets for source-grounded knowledge creation."""

from pydantic import BaseModel, ConfigDict, Field

from kms.config.inference import NoRetryVisionInferenceSettings


class UnitExtractionSettings(BaseModel):
    """Configure vision inference and neighboring unit context."""

    model_config = ConfigDict(extra='forbid')

    inference: NoRetryVisionInferenceSettings = Field(
        default_factory=NoRetryVisionInferenceSettings
    )
    input_token_budget: int = Field(default=16384, gt=0)
    safety_margin_tokens: int = Field(default=256, ge=0)
    before_tokens: int = Field(default=1024, ge=0)
    after_tokens: int = Field(default=1024, ge=0)
