"""Inference budgets for task grading."""

from pydantic import BaseModel, ConfigDict, Field

from kms.config.inference import NoRetryVisionInferenceSettings


class TaskGradingSettings(BaseModel):
    """Configure complete-request token and native-image budgets."""

    model_config = ConfigDict(extra='forbid')

    inference: NoRetryVisionInferenceSettings = Field(
        default_factory=NoRetryVisionInferenceSettings
    )
    input_token_budget: int = Field(default=16384, gt=0)
    safety_margin_tokens: int = Field(default=256, ge=0)
