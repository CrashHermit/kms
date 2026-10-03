"""Typed source-triplet learning contracts."""

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


class SourceTripletLearningFact(_LearningFactVertex):
    """Durable atomic learning fact about one source triplet hub."""


class SourceTripletLearningFactInput(_LearningFactInput):
    """Triplet focus and source-fact text supplied to inference."""


class SourceTripletLearningFactRequest(_LearningFactRequest):
    """Backend request pairing a triplet hub with one source fact."""

    def model_input(self) -> SourceTripletLearningFactInput:
        """Project graph-backed request data to model-facing content."""
        return SourceTripletLearningFactInput(
            hub_name=self.hub_name,
            source_fact_text=self.source_fact_text,
        )


class SourceTripletLearningFactCandidate(_LearningFactCandidate):
    """One candidate triplet learning fact."""


class SourceTripletLearningFactOccurrence(_LearningFactOccurrence):
    """Triplet learning fact with its originating hub and source fact."""

    learning_fact: SourceTripletLearningFact


class SourceTripletFlashcardInput(_FlashcardInput):
    """One triplet learning-fact text supplied to card inference."""


class SourceTripletFlashcardRequest(BaseModel):
    """Persisted triplet learning fact requested for card generation."""

    learning_fact: SourceTripletLearningFact

    def model_input(self) -> SourceTripletFlashcardInput:
        """Project the persisted fact to its model-facing text."""
        return SourceTripletFlashcardInput(
            learning_fact_text=self.learning_fact.text
        )


class SourceTripletFlashcardResult(_FlashcardResult):
    """One semantic triplet flashcard result."""


__all__ = [
    'SourceTripletFlashcardInput',
    'SourceTripletFlashcardRequest',
    'SourceTripletFlashcardResult',
    'SourceTripletLearningFact',
    'SourceTripletLearningFactCandidate',
    'SourceTripletLearningFactInput',
    'SourceTripletLearningFactOccurrence',
    'SourceTripletLearningFactRequest',
]
