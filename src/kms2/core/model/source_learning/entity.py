"""Typed source-entity learning contracts."""

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


class SourceEntityLearningFact(_LearningFactVertex):
    """Durable atomic learning fact about one source entity hub."""


class SourceEntityLearningFactInput(_LearningFactInput):
    """Entity focus and source-fact text supplied to inference."""


class SourceEntityLearningFactRequest(_LearningFactRequest):
    """Backend request pairing an entity hub with one source fact."""

    def model_input(self) -> SourceEntityLearningFactInput:
        """Project graph-backed request data to model-facing content."""
        return SourceEntityLearningFactInput(
            hub_name=self.hub_name,
            source_fact_text=self.source_fact_text,
        )


class SourceEntityLearningFactCandidate(_LearningFactCandidate):
    """One candidate entity learning fact."""


class SourceEntityLearningFactOccurrence(_LearningFactOccurrence):
    """Entity learning fact with its originating hub and source fact."""

    learning_fact: SourceEntityLearningFact


class SourceEntityFlashcardInput(_FlashcardInput):
    """One entity learning-fact text supplied to card inference."""


class SourceEntityFlashcardRequest(BaseModel):
    """Persisted entity learning fact requested for card generation."""

    learning_fact: SourceEntityLearningFact

    def model_input(self) -> SourceEntityFlashcardInput:
        """Project the persisted fact to its model-facing text."""
        return SourceEntityFlashcardInput(
            learning_fact_text=self.learning_fact.text
        )


class SourceEntityFlashcardResult(_FlashcardResult):
    """One semantic entity flashcard result."""


__all__ = [
    'SourceEntityFlashcardInput',
    'SourceEntityFlashcardRequest',
    'SourceEntityFlashcardResult',
    'SourceEntityLearningFact',
    'SourceEntityLearningFactCandidate',
    'SourceEntityLearningFactInput',
    'SourceEntityLearningFactOccurrence',
    'SourceEntityLearningFactRequest',
]
