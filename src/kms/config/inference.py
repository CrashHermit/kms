"""Shared inference and context-window configuration."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class PredictorStrategy(StrEnum):
    """Available DSPy predictor strategies."""

    PREDICT = 'predict'
    CHAIN_OF_THOUGHT = 'chain_of_thought'


class StageInferenceSettings(BaseModel):
    """Request-time settings and an optional named inference profile."""

    model_config = ConfigDict(extra='forbid')

    profile: str | None = Field(default=None, min_length=1)
    strategy: PredictorStrategy = PredictorStrategy.PREDICT
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
    max_tokens: int = 8192
    num_retries: int = 3


class TextInferenceSettings(StageInferenceSettings):
    """Request-time inference settings for text model-server stages."""

    temperature: float = 0.6
    top_k: int = 20
    max_tokens: int = 16384


class VisionInferenceSettings(StageInferenceSettings):
    """Request-time inference settings for vision model-server stages."""

    profile: str | None = Field(default='chatgpt-luna', min_length=1)


class ImagePayloadSettings(BaseModel):
    """Aggregate image limits for one native-image inference request."""

    model_config = ConfigDict(extra='forbid')

    max_images: int = Field(default=4, ge=0)
    max_bytes: int = Field(default=26214400, gt=0)
    max_pixels: int = Field(default=40000000, gt=0)


class NoRetryVisionInferenceSettings(VisionInferenceSettings):
    """Vision-stage inference settings that never retry requests."""

    num_retries: int = 0


class ContextWindowSettings(BaseModel):
    """Model-token budgets for estimated text-plus-image source material."""

    backward_budget: int | None = None
    forward_budget: int | None = None
    target_budget: int = 0
