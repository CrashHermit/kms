"""Contracts for source-local learning facts and flashcards.

Backend requests retain graph identity and project content-only model inputs.
Each learning request pairs one typed hub with one originating source fact;
collection nodes materialize UUID4 vertices and attach that provenance.
Card requests project only persisted learning-fact text. Repositories persist
completed occurrences without allocating replacement identities. Regeneration
clears the source's generated facts, cards, and review history.
"""

from kms2.core.model.source_learning.entity import (
    SourceEntityFlashcardInput,
    SourceEntityFlashcardRequest,
    SourceEntityFlashcardResult,
    SourceEntityLearningFact,
    SourceEntityLearningFactCandidate,
    SourceEntityLearningFactInput,
    SourceEntityLearningFactOccurrence,
    SourceEntityLearningFactRequest,
)
from kms2.core.model.source_learning.event import (
    SourceEventFlashcardInput,
    SourceEventFlashcardRequest,
    SourceEventFlashcardResult,
    SourceEventLearningFact,
    SourceEventLearningFactCandidate,
    SourceEventLearningFactInput,
    SourceEventLearningFactOccurrence,
    SourceEventLearningFactRequest,
)
from kms2.core.model.source_learning.flashcard import (
    SourceFlashcard,
    SourceFlashcardOccurrence,
)
from kms2.core.model.source_learning.predicate import (
    SourcePredicateFlashcardInput,
    SourcePredicateFlashcardRequest,
    SourcePredicateFlashcardResult,
    SourcePredicateLearningFact,
    SourcePredicateLearningFactCandidate,
    SourcePredicateLearningFactInput,
    SourcePredicateLearningFactOccurrence,
    SourcePredicateLearningFactRequest,
)
from kms2.core.model.source_learning.triplet import (
    SourceTripletFlashcardInput,
    SourceTripletFlashcardRequest,
    SourceTripletFlashcardResult,
    SourceTripletLearningFact,
    SourceTripletLearningFactCandidate,
    SourceTripletLearningFactInput,
    SourceTripletLearningFactOccurrence,
    SourceTripletLearningFactRequest,
)

__all__ = [
    'SourceEntityFlashcardInput',
    'SourceEntityFlashcardRequest',
    'SourceEntityFlashcardResult',
    'SourceEntityLearningFact',
    'SourceEntityLearningFactCandidate',
    'SourceEntityLearningFactInput',
    'SourceEntityLearningFactOccurrence',
    'SourceEntityLearningFactRequest',
    'SourceEventFlashcardInput',
    'SourceEventFlashcardRequest',
    'SourceEventFlashcardResult',
    'SourceEventLearningFact',
    'SourceEventLearningFactCandidate',
    'SourceEventLearningFactInput',
    'SourceEventLearningFactOccurrence',
    'SourceEventLearningFactRequest',
    'SourceFlashcard',
    'SourceFlashcardOccurrence',
    'SourcePredicateFlashcardInput',
    'SourcePredicateFlashcardRequest',
    'SourcePredicateFlashcardResult',
    'SourcePredicateLearningFact',
    'SourcePredicateLearningFactCandidate',
    'SourcePredicateLearningFactInput',
    'SourcePredicateLearningFactOccurrence',
    'SourcePredicateLearningFactRequest',
    'SourceTripletFlashcardInput',
    'SourceTripletFlashcardRequest',
    'SourceTripletFlashcardResult',
    'SourceTripletLearningFact',
    'SourceTripletLearningFactCandidate',
    'SourceTripletLearningFactInput',
    'SourceTripletLearningFactOccurrence',
    'SourceTripletLearningFactRequest',
]
