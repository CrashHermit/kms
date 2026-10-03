"""Typed source-event learning contracts."""

from pydantic import BaseModel

from kms2.core.model.source_learning._base import (
    _FlashcardInput,
    _FlashcardResult,
    _LearningFactCandidate,
    _LearningFactInput,
    _LearningFactOccurrence,
    _LearningFactRequest,
    _LearningFactVertex,
)


class SourceEventLearningFact(_LearningFactVertex):
    """Durable atomic learning fact about one source event hub."""


class SourceEventLearningFactInput(_LearningFactInput):
    """Event focus and source-fact text supplied to inference."""


class SourceEventLearningFactRequest(_LearningFactRequest):
    """Backend request pairing an event hub with one source fact."""

    def model_input(self) -> SourceEventLearningFactInput:
        """Project graph-backed request data to model-facing content."""
        return SourceEventLearningFactInput(
            hub_name=self.hub_name,
            source_fact_text=self.source_fact_text,
        )


class SourceEventLearningFactCandidate(_LearningFactCandidate):
    """One candidate event learning fact."""


class SourceEventLearningFactOccurrence(_LearningFactOccurrence):
    """Event learning fact with its originating hub and source fact."""

    learning_fact: SourceEventLearningFact


class SourceEventFlashcardInput(_FlashcardInput):
    """One event learning-fact text supplied to card inference."""


class SourceEventFlashcardRequest(BaseModel):
    """Persisted event learning fact requested for card generation."""

    learning_fact: SourceEventLearningFact

    def model_input(self) -> SourceEventFlashcardInput:
        """Project the persisted fact to its model-facing text."""
        return SourceEventFlashcardInput(
            learning_fact_text=self.learning_fact.text
        )


class SourceEventFlashcardResult(_FlashcardResult):
    """One semantic event flashcard result."""


__all__ = [
    'SourceEventFlashcardInput',
    'SourceEventFlashcardRequest',
    'SourceEventFlashcardResult',
    'SourceEventLearningFact',
    'SourceEventLearningFactCandidate',
    'SourceEventLearningFactInput',
    'SourceEventLearningFactOccurrence',
    'SourceEventLearningFactRequest',
]
