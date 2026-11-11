"""Settings for instruction governance."""

from pydantic import BaseModel, Field

from kms.config.inference import (
    ContextWindowSettings,
    ImagePayloadSettings,
    VisionInferenceSettings,
)


class InstructionGovernanceSettings(BaseModel):
    """Language model and context-window settings for instruction governance."""

    inference: VisionInferenceSettings = Field(
        default_factory=VisionInferenceSettings
    )
    batch_size: int = Field(default=16, gt=0)
    input_token_budget: int = Field(default=4096, gt=0)
    safety_margin_tokens: int = Field(default=256, ge=0)
    context_window: ContextWindowSettings = Field(
        default_factory=lambda: ContextWindowSettings(
            backward_budget=200, forward_budget=500
        )
    )
    image_payload: ImagePayloadSettings = Field(
        default_factory=ImagePayloadSettings
    )
