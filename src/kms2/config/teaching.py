"""Configuration for the goal-driven teaching workflow."""

from pydantic import BaseModel, Field

from kms2.config.inference import VisionInferenceSettings
from kms2.config.source_search import SourceSearchSettings


class TeachingSettings(BaseModel):
    """Inference, payload, and tool limits for one teacher invocation."""

    inference: VisionInferenceSettings = Field(
        default_factory=lambda: VisionInferenceSettings(num_retries=0)
    )
    input_token_budget: int = Field(default=16384, gt=0)
    safety_margin_tokens: int = Field(default=256, ge=0)
    max_tool_calls: int = Field(default=8, ge=0)
    max_context_images: int = Field(default=4, ge=0)
    max_context_image_bytes: int = Field(default=26214400, gt=0)
    max_context_image_pixels: int = Field(default=40000000, gt=0)
    discovery_preview_characters: int = Field(default=320, gt=0)
    evidence_catalog_page_size: int = Field(default=12, gt=0)
    learner_history: SourceSearchSettings = Field(
        default_factory=SourceSearchSettings
    )
