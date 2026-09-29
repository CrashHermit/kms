"""LangGraph state for typed source-learning phases."""

import operator
from typing import Annotated

from pydantic import BaseModel, Field

from kms2.core.model.source_learning.entity import (
    SourceEntityFlashcardInput,
    SourceEntityFlashcardResult,
    SourceEntityLearningFactInput,
    SourceEntityLearningFactResult,
)
from kms2.core.model.source_learning.event import (
    SourceEventFlashcardInput,
    SourceEventFlashcardResult,
    SourceEventLearningFactInput,
    SourceEventLearningFactResult,
)
from kms2.core.model.source_learning.predicate import (
    SourcePredicateFlashcardInput,
    SourcePredicateFlashcardResult,
    SourcePredicateLearningFactInput,
    SourcePredicateLearningFactResult,
)
from kms2.core.model.source_learning.triplet import (
    SourceTripletFlashcardInput,
    SourceTripletFlashcardResult,
    SourceTripletLearningFactInput,
    SourceTripletLearningFactResult,
)


class SourceEntityLearningFactWorkerResult(BaseModel):
    """One ordered entity learning-fact worker result."""

    ordinal: int = Field(ge=0)
    result: SourceEntityLearningFactResult


class SourceEventLearningFactWorkerResult(BaseModel):
    """One ordered event learning-fact worker result."""

    ordinal: int = Field(ge=0)
    result: SourceEventLearningFactResult


class SourcePredicateLearningFactWorkerResult(BaseModel):
    """One ordered predicate learning-fact worker result."""

    ordinal: int = Field(ge=0)
    result: SourcePredicateLearningFactResult


class SourceTripletLearningFactWorkerResult(BaseModel):
    """One ordered triplet learning-fact worker result."""

    ordinal: int = Field(ge=0)
    result: SourceTripletLearningFactResult


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
    entity_learning_fact_inputs: list[SourceEntityLearningFactInput] = Field(
        default_factory=list
    )
    entity_learning_fact_results: Annotated[
        list[SourceEntityLearningFactWorkerResult], operator.add
    ] = Field(default_factory=list)
    entity_learning_fact_results_ordered: list[
        SourceEntityLearningFactResult
    ] = Field(default_factory=list)
    entity_learning_fact_count: int = 0
    entity_flashcard_inputs: list[SourceEntityFlashcardInput] = Field(
        default_factory=list
    )
    entity_flashcard_results: Annotated[
        list[SourceEntityFlashcardWorkerResult], operator.add
    ] = Field(default_factory=list)
    entity_flashcard_results_ordered: list[SourceEntityFlashcardResult] = Field(
        default_factory=list
    )
    entity_flashcard_count: int = 0
    event_learning_fact_inputs: list[SourceEventLearningFactInput] = Field(
        default_factory=list
    )
    event_learning_fact_results: Annotated[
        list[SourceEventLearningFactWorkerResult], operator.add
    ] = Field(default_factory=list)
    event_learning_fact_results_ordered: list[SourceEventLearningFactResult] = (
        Field(default_factory=list)
    )
    event_learning_fact_count: int = 0
    event_flashcard_inputs: list[SourceEventFlashcardInput] = Field(
        default_factory=list
    )
    event_flashcard_results: Annotated[
        list[SourceEventFlashcardWorkerResult], operator.add
    ] = Field(default_factory=list)
    event_flashcard_results_ordered: list[SourceEventFlashcardResult] = Field(
        default_factory=list
    )
    event_flashcard_count: int = 0
    predicate_learning_fact_inputs: list[SourcePredicateLearningFactInput] = (
        Field(default_factory=list)
    )
    predicate_learning_fact_results: Annotated[
        list[SourcePredicateLearningFactWorkerResult], operator.add
    ] = Field(default_factory=list)
    predicate_learning_fact_results_ordered: list[
        SourcePredicateLearningFactResult
    ] = Field(default_factory=list)
    predicate_learning_fact_count: int = 0
    predicate_flashcard_inputs: list[SourcePredicateFlashcardInput] = Field(
        default_factory=list
    )
    predicate_flashcard_results: Annotated[
        list[SourcePredicateFlashcardWorkerResult], operator.add
    ] = Field(default_factory=list)
    predicate_flashcard_results_ordered: list[
        SourcePredicateFlashcardResult
    ] = Field(default_factory=list)
    predicate_flashcard_count: int = 0
    triplet_learning_fact_inputs: list[SourceTripletLearningFactInput] = Field(
        default_factory=list
    )
    triplet_learning_fact_results: Annotated[
        list[SourceTripletLearningFactWorkerResult], operator.add
    ] = Field(default_factory=list)
    triplet_learning_fact_results_ordered: list[
        SourceTripletLearningFactResult
    ] = Field(default_factory=list)
    triplet_learning_fact_count: int = 0
    triplet_flashcard_inputs: list[SourceTripletFlashcardInput] = Field(
        default_factory=list
    )
    triplet_flashcard_results: Annotated[
        list[SourceTripletFlashcardWorkerResult], operator.add
    ] = Field(default_factory=list)
    triplet_flashcard_results_ordered: list[SourceTripletFlashcardResult] = (
        Field(default_factory=list)
    )
    triplet_flashcard_count: int = 0
