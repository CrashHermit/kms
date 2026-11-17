"""Vision inference settings for source-grounded correctness judgment."""

from pydantic import BaseModel, ConfigDict, Field

from kms.config.inference import VisionInferenceSettings


class CorrectnessSettings(BaseModel):
    """Configure complete-request token and native-image budgets."""

    model_config = ConfigDict(extra='forbid')

    inference: VisionInferenceSettings = Field(
        default_factory=lambda: VisionInferenceSettings(num_retries=0)
    )
    input_token_budget: int = Field(default=16384, gt=0)
    safety_margin_tokens: int = Field(default=256, ge=0)
