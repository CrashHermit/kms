from pydantic import BaseModel, Field

from kms.config.inference import VisionInferenceSettings


class ImageSeamSettings(BaseModel):
    """Vision inference and text-metadata batching for image seams."""

    inference: VisionInferenceSettings = Field(
        default_factory=VisionInferenceSettings
    )
    batch_size: int = Field(default=4, gt=0)
    input_token_budget: int = Field(default=2048, gt=0)
    safety_margin_tokens: int = Field(default=256, ge=0)
