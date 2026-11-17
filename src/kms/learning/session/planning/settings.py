"""Inference and request-budget settings for learning planning."""

from pydantic import BaseModel, ConfigDict, Field

from kms.config.inference import NoRetryVisionInferenceSettings


class LearningPlanningSettings(BaseModel):
    """Configure no-retry multimodal planning inference."""

    model_config = ConfigDict(extra='forbid')

    inference: NoRetryVisionInferenceSettings = Field(
        default_factory=NoRetryVisionInferenceSettings
    )
    input_token_budget: int = Field(default=16384, gt=0)
    safety_margin_tokens: int = Field(default=256, ge=0)
