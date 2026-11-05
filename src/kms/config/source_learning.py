"""Configuration for the two source-learning inference passes."""

from pydantic import BaseModel, Field

from kms.config.inference import (
    ContextWindowSettings,
    ImagePayloadSettings,
    NoRetryTextInferenceSettings,
)


def _default_flashcard_inference() -> NoRetryTextInferenceSettings:
    """Use the selected text backend with a compact completion reserve."""
    return NoRetryTextInferenceSettings(max_tokens=2048)


class SourceLearningSettings(BaseModel):
    """Independent inference profiles and payload budgets for source learning."""

    context_window: ContextWindowSettings = Field(
        default_factory=lambda: ContextWindowSettings(
            backward_budget=400,
            forward_budget=400,
        )
    )
    atomic_flashcards: NoRetryTextInferenceSettings = Field(
        default_factory=_default_flashcard_inference
    )
    coherent_flashcards: NoRetryTextInferenceSettings = Field(
        default_factory=_default_flashcard_inference
    )
    image_payload: ImagePayloadSettings = Field(
        default_factory=ImagePayloadSettings
    )
    input_token_budget: int = Field(default=8192, gt=0)
    safety_margin_tokens: int = Field(default=256, ge=0)
