"""LangGraph state for the two bounded source-learning passes."""

import operator
from typing import Annotated

from pydantic import BaseModel, Field

from kms2.core.model.source_learning.atomic import SourceAtomicFlashcardRequest
from kms2.core.model.source_learning.coherent import (
    SourceAtomicFlashcardPacket,
    SourceCoherentFlashcardCandidate,
    SourceCoherentFlashcardRequest,
)
from kms2.core.model.source_learning.flashcard import (
    SourceFlashcardCandidate,
    SourceFlashcardOccurrence,
)


class SourceAtomicFlashcardWorkerResult(BaseModel):
    """One ordered atomic flashcard worker result."""

    ordinal: int = Field(ge=0)
    cards: list[SourceFlashcardCandidate] = Field(default_factory=list)


class SourceCoherentFlashcardWorkerResult(BaseModel):
    """One ordered coherent flashcard worker result."""

    ordinal: int = Field(ge=0)
    cards: list[SourceCoherentFlashcardCandidate] = Field(default_factory=list)


class SourceLearningState(BaseModel):
    """State shared by cleanup, atomic generation, and coherent generation."""

    source_uuid: str
    atomic_requests: list[SourceAtomicFlashcardRequest] = Field(
        default_factory=list
    )
    atomic_results: Annotated[
        list[SourceAtomicFlashcardWorkerResult], operator.add
    ] = Field(default_factory=list)
    atomic_packets: list[SourceAtomicFlashcardPacket] = Field(
        default_factory=list
    )
    coherent_requests: list[SourceCoherentFlashcardRequest] = Field(
        default_factory=list
    )
    coherent_results: Annotated[
        list[SourceCoherentFlashcardWorkerResult], operator.add
    ] = Field(default_factory=list)
    coherent_occurrences: list[SourceFlashcardOccurrence] = Field(
        default_factory=list
    )
    atomic_flashcard_count: int = 0
    coherent_flashcard_count: int = 0
