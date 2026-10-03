"""Typed source-predicate learning contracts."""

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


class SourcePredicateLearningFact(_LearningFactVertex):
    """Durable atomic learning fact about one source predicate hub."""


class SourcePredicateLearningFactInput(_LearningFactInput):
    """Predicate focus and source-fact text supplied to inference."""


class SourcePredicateLearningFactRequest(_LearningFactRequest):
    """Backend request pairing a predicate hub with one source fact."""

    def model_input(self) -> SourcePredicateLearningFactInput:
        """Project graph-backed request data to model-facing content."""
        return SourcePredicateLearningFactInput(
            hub_name=self.hub_name,
            source_fact_text=self.source_fact_text,
        )


class SourcePredicateLearningFactCandidate(_LearningFactCandidate):
    """One candidate predicate learning fact."""


class SourcePredicateLearningFactOccurrence(_LearningFactOccurrence):
    """Predicate learning fact with its originating hub and source fact."""

    learning_fact: SourcePredicateLearningFact


class SourcePredicateFlashcardInput(_FlashcardInput):
    """One predicate learning-fact text supplied to card inference."""


class SourcePredicateFlashcardRequest(BaseModel):
    """Persisted predicate learning fact requested for card generation."""

    learning_fact: SourcePredicateLearningFact

    def model_input(self) -> SourcePredicateFlashcardInput:
        """Project the persisted fact to its model-facing text."""
        return SourcePredicateFlashcardInput(
            learning_fact_text=self.learning_fact.text
        )


class SourcePredicateFlashcardResult(_FlashcardResult):
    """One semantic predicate flashcard result."""


__all__ = [
    'SourcePredicateFlashcardInput',
    'SourcePredicateFlashcardRequest',
    'SourcePredicateFlashcardResult',
    'SourcePredicateLearningFact',
    'SourcePredicateLearningFactCandidate',
    'SourcePredicateLearningFactInput',
    'SourcePredicateLearningFactOccurrence',
    'SourcePredicateLearningFactRequest',
]
