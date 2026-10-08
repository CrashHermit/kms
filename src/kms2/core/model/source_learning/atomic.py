"""Backend and model-facing contracts for atomic flashcard generation."""

from pydantic import BaseModel, Field

from kms2.core.model.source_learning.context import (
    SourceLearningEvidence,
    SourceLearningHubContext,
)
from kms2.core.model.source_learning.flashcard import SourceFlashcardCandidate


class SourceAtomicFlashcardInput(BaseModel):
    """Content supplied to one atomic flashcard inference request."""

    context: SourceLearningHubContext
    evidence: SourceLearningEvidence


class SourceAtomicFlashcardRequest(BaseModel):
    """Backend request retaining provenance for one atomic inference call."""

    hub_uuid: str = Field(min_length=1)
    triplet_uuid: str = Field(min_length=1)
    source_fact_uuid: str = Field(min_length=1)
    input: SourceAtomicFlashcardInput

    def model_input(self) -> SourceAtomicFlashcardInput:
        """Project the backend request to content-only model input."""
        return self.input


__all__ = [
    'SourceAtomicFlashcardInput',
    'SourceAtomicFlashcardRequest',
    'SourceFlashcardCandidate',
]
