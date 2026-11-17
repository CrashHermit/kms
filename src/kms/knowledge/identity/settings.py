"""Inference and capacity settings for proposition identity auditing."""

from pydantic import BaseModel, ConfigDict, Field

from kms.config.inference import TextInferenceSettings


class IdentitySettings(BaseModel):
    """Configure proposition identity matching and request capacity."""

    model_config = ConfigDict(extra='forbid')

    inference: TextInferenceSettings = Field(
        default_factory=lambda: TextInferenceSettings(num_retries=0)
    )
    candidate_limit: int = Field(default=16, gt=0)
    input_token_budget: int = Field(default=8192, gt=0)
    safety_margin_tokens: int = Field(default=256, ge=0)
