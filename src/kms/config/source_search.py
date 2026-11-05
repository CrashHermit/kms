"""Configuration for user-scoped source retrieval and final search selection."""

from pydantic import BaseModel, ConfigDict, Field

from kms.config.inference import ImagePayloadSettings, VisionInferenceSettings


class SourceSearchSettings(BaseModel):
    """Candidate and final-result limits for source search."""

    candidate_limit: int = Field(default=32, gt=0)
    result_limit: int = Field(default=8, gt=0)


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
