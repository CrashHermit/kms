from pydantic import BaseModel, Field

from kms.config.inference import (
    ContextWindowSettings,
    ImagePayloadSettings,
    VisionInferenceSettings,
)


class ExerciseSplitterSettings(BaseModel):
    """Language model and context-window settings for exercise splitting."""

    router: VisionInferenceSettings = Field(
        default_factory=VisionInferenceSettings
    )
    splitter: VisionInferenceSettings = Field(
        default_factory=VisionInferenceSettings
    )
    batch_size: int = Field(default=32, gt=0)
    context_window: ContextWindowSettings = Field(
        default_factory=lambda: ContextWindowSettings(target_budget=2048)
    )
    image_payload: ImagePayloadSettings = Field(
        default_factory=lambda: ImagePayloadSettings(max_images=6)
    )
    router_input_token_budget: int = Field(default=8192, gt=0)
    splitter_input_token_budget: int = Field(default=8192, gt=0)
    safety_margin_tokens: int = Field(default=256, ge=0)
