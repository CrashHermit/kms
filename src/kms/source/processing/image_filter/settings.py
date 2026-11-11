from pydantic import BaseModel, Field

from kms.config.inference import (
    ContextWindowSettings,
    ImagePayloadSettings,
    VisionInferenceSettings,
)


class ImageFilterSettings(BaseModel):
    """Vision inference and context settings for image retention decisions."""

    inference: VisionInferenceSettings = Field(
        default_factory=lambda: VisionInferenceSettings(max_tokens=1024)
    )
    context_window: ContextWindowSettings = Field(
        default_factory=lambda: ContextWindowSettings(
            backward_budget=1024, forward_budget=1024
        )
    )
    image_payload: ImagePayloadSettings = Field(
        default_factory=lambda: ImagePayloadSettings(max_images=9)
    )
    batch_size: int = Field(default=4, gt=0)
    input_token_budget: int = Field(default=2048, gt=0)
    safety_margin_tokens: int = Field(default=256, ge=0)
