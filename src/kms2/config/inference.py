"""Shared inference and context-window configuration."""

from enum import StrEnum

from pydantic import BaseModel


class PredictorStrategy(StrEnum):
    """Available DSPy predictor strategies."""

    PREDICT = 'predict'
    CHAIN_OF_THOUGHT = 'chain_of_thought'


class StageInferenceSettings(BaseModel):
    """Request-time inference settings for one source stage."""

    model_server_profile: str
    strategy: PredictorStrategy = PredictorStrategy.PREDICT
    temperature: float = 0.0
    max_tokens: int = 8192
    num_retries: int = 3


class TextInferenceSettings(StageInferenceSettings):
    """Request-time inference settings for text model-server stages."""

    model_server_profile: str = 'gemma-text-32k'


class VisionInferenceSettings(StageInferenceSettings):
    """Request-time inference settings for vision model-server stages."""

    model_server_profile: str = 'gemma-vision-8k'


class NoRetryTextInferenceSettings(TextInferenceSettings):
    """Text-stage inference settings that never retry requests."""

    num_retries: int = 0


class ContextWindowSettings(BaseModel):
    """Estimated token budgets for surrounding source context."""

    backward_budget: int | None = None
    forward_budget: int | None = None
    target_budget: int = 0
