"""Shared inference and context-window configuration."""

from enum import StrEnum

from pydantic import BaseModel, Field


class PredictorStrategy(StrEnum):
    """Available DSPy predictor strategies."""

    PREDICT = 'predict'
    CHAIN_OF_THOUGHT = 'chain_of_thought'


class StageInferenceSettings(BaseModel):
    """Request-time inference settings for one source stage."""

    model_server_profile: str
    strategy: PredictorStrategy = PredictorStrategy.PREDICT
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
    max_tokens: int = 8192
    num_retries: int = 3


class TextInferenceSettings(StageInferenceSettings):
    """Request-time inference settings for text model-server stages."""

    model_server_profile: str = 'qwen3.8-9b-distill-text'
    temperature: float = 0.6
    top_k: int = 20
    max_tokens: int = 16384


class VisionInferenceSettings(StageInferenceSettings):
    """Request-time inference settings for vision model-server stages."""

    model_server_profile: str = 'gemma-vision-8k'


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
    """Model-token budgets for surrounding source text content."""

    backward_budget: int | None = None
    forward_budget: int | None = None
    target_budget: int = 0
