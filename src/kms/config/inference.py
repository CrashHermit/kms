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


class NoRetryTextInferenceSettings(TextInferenceSettings):
    """Text-stage inference settings that never retry requests."""

    num_retries: int = 0


class HubSynthesisBudgetSettings(BaseModel):
    """Estimated payload budgets for synthesis and temporary summaries.

    ``input_token_budget`` caps the serialized application payload. The
    configured model capacity also reserves the stage's completion limit and
    safety margin.
    ``summary_completion_tokens`` reserves leaf and merge completions without
    changing final synthesis inference settings.
    Leaf and merge stages reuse the final stage's profile, strategy, and
    sampling settings.

    Valid budgets fit each whole evidence record in a leaf request and at
    least two summaries in a merge request. Records and summaries are
    indivisible; oversized records are input-budget errors. Workers do not
    validate, repair, or retry outputs.
    """

    input_token_budget: int = Field(default=8192, gt=0)
    summary_completion_tokens: int = Field(default=4096, gt=0)
    safety_margin_tokens: int = Field(default=256, ge=0)


class ContextWindowSettings(BaseModel):
    """Model-token budgets for estimated text-plus-image source material."""

    backward_budget: int | None = None
    forward_budget: int | None = None
    target_budget: int = 0


class TermDescriptionSettings(BaseModel):
    """Language model and context-window settings for typed descriptions."""

    inference: NoRetryTextInferenceSettings = Field(
        default_factory=NoRetryTextInferenceSettings
    )
    context_window: ContextWindowSettings = Field(
        default_factory=lambda: ContextWindowSettings(
            backward_budget=400,
            forward_budget=400,
        )
    )
    image_payload: ImagePayloadSettings = Field(
        default_factory=ImagePayloadSettings
    )
    batch_size: int = Field(default=16, gt=0)
    input_token_budget: int = Field(default=4096, gt=0)
    safety_margin_tokens: int = Field(default=256, ge=0)
