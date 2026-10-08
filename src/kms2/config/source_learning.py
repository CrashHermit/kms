"""Configuration for the two source-learning inference passes."""

from pydantic import BaseModel, Field

from kms2.config.inference import NoRetryTextInferenceSettings


def _default_flashcard_inference() -> NoRetryTextInferenceSettings:
    """Use the local Gemma E4B profile for source-learning cards."""
    return NoRetryTextInferenceSettings(
        model_server_profile='gemma-text-32k',
        max_tokens=2048,
    )


class SourceLearningSettings(BaseModel):
    """Independent inference profiles and payload budgets for source learning."""

    atomic_flashcards: NoRetryTextInferenceSettings = Field(
        default_factory=_default_flashcard_inference
    )
    coherent_flashcards: NoRetryTextInferenceSettings = Field(
        default_factory=_default_flashcard_inference
    )
    input_token_budget: int = Field(default=8192, gt=0)
    safety_margin_tokens: int = Field(default=256, ge=0)
