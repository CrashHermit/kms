"""Configuration for final search selection."""

from pydantic import BaseModel, ConfigDict, Field

from kms.config.inference import ImagePayloadSettings, VisionInferenceSettings


class SearchSelectionSettings(BaseModel):
    """Bounded inference and image payload for final search selection."""

    model_config = ConfigDict(extra='forbid')

    inference: VisionInferenceSettings = Field(
        default_factory=lambda: VisionInferenceSettings(
            max_tokens=1024,
            num_retries=0,
        )
    )
    input_token_budget: int = Field(default=16384, gt=0)
    safety_margin_tokens: int = Field(default=256, ge=0)
    image_payload: ImagePayloadSettings = Field(
        default_factory=ImagePayloadSettings
    )
