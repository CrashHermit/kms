"""Settings for pedagogical discovery."""

from pydantic import BaseModel, Field

from kms.config.inference import (
    ContextWindowSettings,
    ImagePayloadSettings,
    VisionInferenceSettings,
)


class PedagogicalFinderSettings(BaseModel):
    """Language model and context-window settings for pedagogical discovery."""

    start_router: VisionInferenceSettings = Field(
        default_factory=VisionInferenceSettings
    )
    boundary_router: VisionInferenceSettings = Field(
        default_factory=VisionInferenceSettings
    )
    start_batch_size: int = Field(default=16, gt=0)
    boundary_batch_size: int = Field(default=16, gt=0)
    start_input_token_budget: int = Field(default=4096, gt=0)
    boundary_input_token_budget: int = Field(default=4096, gt=0)
    safety_margin_tokens: int = Field(default=256, ge=0)
    start_context_window: ContextWindowSettings = Field(
        default_factory=lambda: ContextWindowSettings(
            backward_budget=300, forward_budget=300
        )
    )
    boundary_context_window: ContextWindowSettings = Field(
        default_factory=lambda: ContextWindowSettings(
            backward_budget=300, forward_budget=300
        )
    )
    image_payload: ImagePayloadSettings = Field(
        default_factory=ImagePayloadSettings
    )
