"""LangGraph state for typed source-learning phases."""

import operator
from typing import Annotated

from pydantic import BaseModel, Field

from kms2.core.model.source_learning.entity import (
    SourceEntityFlashcardRequest,
    SourceEntityFlashcardResult,
    SourceEntityLearningFactCandidate,
    SourceEntityLearningFactOccurrence,
    SourceEntityLearningFactRequest,
)
from kms2.core.model.source_learning.event import (
    SourceEventFlashcardRequest,
    SourceEventFlashcardResult,
    SourceEventLearningFactCandidate,
    SourceEventLearningFactOccurrence,
    SourceEventLearningFactRequest,
)
from kms2.core.model.source_learning.flashcard import SourceFlashcardOccurrence
from kms2.core.model.source_learning.predicate import (
    SourcePredicateFlashcardRequest,
    SourcePredicateFlashcardResult,
    SourcePredicateLearningFactCandidate,
    SourcePredicateLearningFactOccurrence,
    SourcePredicateLearningFactRequest,
)
from kms2.core.model.source_learning.triplet import (
    SourceTripletFlashcardRequest,
    SourceTripletFlashcardResult,
    SourceTripletLearningFactCandidate,
    SourceTripletLearningFactOccurrence,
    SourceTripletLearningFactRequest,
)


class SourceEntityLearningFactWorkerResult(BaseModel):
    """One ordered entity learning-fact worker result."""

    ordinal: int = Field(ge=0)
    facts: list[SourceEntityLearningFactCandidate] = Field(default_factory=list)


class SourceEventLearningFactWorkerResult(BaseModel):
    """One ordered event learning-fact worker result."""

    ordinal: int = Field(ge=0)
    facts: list[SourceEventLearningFactCandidate] = Field(default_factory=list)


class SourcePredicateLearningFactWorkerResult(BaseModel):
    """One ordered predicate learning-fact worker result."""

    ordinal: int = Field(ge=0)
    facts: list[SourcePredicateLearningFactCandidate] = Field(
        default_factory=list
    )


class SourceTripletLearningFactWorkerResult(BaseModel):
    """One ordered triplet learning-fact worker result."""

    ordinal: int = Field(ge=0)
    facts: list[SourceTripletLearningFactCandidate] = Field(
        default_factory=list
    )


class SourceEntityFlashcardWorkerResult(BaseModel):
    """One ordered entity card worker result."""

    ordinal: int = Field(ge=0)
    result: SourceEntityFlashcardResult


class SourceEventFlashcardWorkerResult(BaseModel):
    """One ordered event card worker result."""

    ordinal: int = Field(ge=0)
    result: SourceEventFlashcardResult


class SourcePredicateFlashcardWorkerResult(BaseModel):
    """One ordered predicate card worker result."""

    ordinal: int = Field(ge=0)
    result: SourcePredicateFlashcardResult


class SourceTripletFlashcardWorkerResult(BaseModel):
    """One ordered triplet card worker result."""

    ordinal: int = Field(ge=0)
    result: SourceTripletFlashcardResult


class SourceLearningState(BaseModel):
    """State shared by the explicit typed source-learning phases."""

    source_uuid: str
    entity_learning_fact_requests: list[SourceEntityLearningFactRequest] = (
        Field(default_factory=list)
    )
    entity_learning_fact_results: Annotated[
        list[SourceEntityLearningFactWorkerResult], operator.add
    ] = Field(default_factory=list)
    entity_learning_fact_occurrences: list[
        SourceEntityLearningFactOccurrence
    ] = Field(default_factory=list)
    entity_learning_fact_count: int = 0
    entity_flashcard_requests: list[SourceEntityFlashcardRequest] = Field(
        default_factory=list
    )
    entity_flashcard_results: Annotated[
        list[SourceEntityFlashcardWorkerResult], operator.add
    ] = Field(default_factory=list)
    entity_flashcard_occurrences: list[SourceFlashcardOccurrence] = Field(
        default_factory=list
    )
    entity_flashcard_count: int = 0
    event_learning_fact_requests: list[SourceEventLearningFactRequest] = Field(
        default_factory=list
    )
    event_learning_fact_results: Annotated[
        list[SourceEventLearningFactWorkerResult], operator.add
    ] = Field(default_factory=list)
    event_learning_fact_occurrences: list[SourceEventLearningFactOccurrence] = (
        Field(default_factory=list)
    )
    event_learning_fact_count: int = 0
    event_flashcard_requests: list[SourceEventFlashcardRequest] = Field(
        default_factory=list
    )
    event_flashcard_results: Annotated[
        list[SourceEventFlashcardWorkerResult], operator.add
    ] = Field(default_factory=list)
    event_flashcard_occurrences: list[SourceFlashcardOccurrence] = Field(
        default_factory=list
    )
    event_flashcard_count: int = 0
    predicate_learning_fact_requests: list[
        SourcePredicateLearningFactRequest
    ] = Field(default_factory=list)
    predicate_learning_fact_results: Annotated[
        list[SourcePredicateLearningFactWorkerResult], operator.add
    ] = Field(default_factory=list)
    predicate_learning_fact_occurrences: list[
        SourcePredicateLearningFactOccurrence
    ] = Field(default_factory=list)
    predicate_learning_fact_count: int = 0
    predicate_flashcard_requests: list[SourcePredicateFlashcardRequest] = Field(
        default_factory=list
    )
    predicate_flashcard_results: Annotated[
        list[SourcePredicateFlashcardWorkerResult], operator.add
    ] = Field(default_factory=list)
    predicate_flashcard_occurrences: list[SourceFlashcardOccurrence] = Field(
        default_factory=list
    )
    predicate_flashcard_count: int = 0
    triplet_learning_fact_requests: list[SourceTripletLearningFactRequest] = (
        Field(default_factory=list)
    )
    triplet_learning_fact_results: Annotated[
        list[SourceTripletLearningFactWorkerResult], operator.add
    ] = Field(default_factory=list)
    triplet_learning_fact_occurrences: list[
        SourceTripletLearningFactOccurrence
    ] = Field(default_factory=list)
    triplet_learning_fact_count: int = 0
    triplet_flashcard_requests: list[SourceTripletFlashcardRequest] = Field(
        default_factory=list
    )
    triplet_flashcard_results: Annotated[
        list[SourceTripletFlashcardWorkerResult], operator.add
    ] = Field(default_factory=list)
    triplet_flashcard_occurrences: list[SourceFlashcardOccurrence] = Field(
        default_factory=list
    )
    triplet_flashcard_count: int = 0
