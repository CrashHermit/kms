"""Contracts for source-local learning facts and flashcards."""

from kms2.core.model.source_learning.entity import (
    SourceEntityFlashcardDraft,
    SourceEntityFlashcardInput,
    SourceEntityFlashcardResult,
    SourceEntityLearningFact,
    SourceEntityLearningFactEvidence,
    SourceEntityLearningFactInput,
    SourceEntityLearningFactResult,
)
from kms2.core.model.source_learning.event import (
    SourceEventFlashcardDraft,
    SourceEventFlashcardInput,
    SourceEventFlashcardResult,
    SourceEventLearningFact,
    SourceEventLearningFactEvidence,
    SourceEventLearningFactInput,
    SourceEventLearningFactResult,
)
from kms2.core.model.source_learning.flashcard import SourceFlashcard
from kms2.core.model.source_learning.predicate import (
    SourcePredicateFlashcardDraft,
    SourcePredicateFlashcardInput,
    SourcePredicateFlashcardResult,
    SourcePredicateLearningFact,
    SourcePredicateLearningFactEvidence,
    SourcePredicateLearningFactInput,
    SourcePredicateLearningFactResult,
)
from kms2.core.model.source_learning.triplet import (
    SourceTripletFlashcardDraft,
    SourceTripletFlashcardInput,
    SourceTripletFlashcardResult,
    SourceTripletLearningFact,
    SourceTripletLearningFactEvidence,
    SourceTripletLearningFactInput,
    SourceTripletLearningFactResult,
)

__all__ = [
    'SourceEntityFlashcardDraft',
    'SourceEntityFlashcardInput',
    'SourceEntityFlashcardResult',
    'SourceEntityLearningFact',
    'SourceEntityLearningFactEvidence',
    'SourceEntityLearningFactInput',
    'SourceEntityLearningFactResult',
    'SourceEventFlashcardDraft',
    'SourceEventFlashcardInput',
    'SourceEventFlashcardResult',
    'SourceEventLearningFact',
    'SourceEventLearningFactEvidence',
    'SourceEventLearningFactInput',
    'SourceEventLearningFactResult',
    'SourceFlashcard',
    'SourcePredicateFlashcardDraft',
    'SourcePredicateFlashcardInput',
    'SourcePredicateFlashcardResult',
    'SourcePredicateLearningFact',
    'SourcePredicateLearningFactEvidence',
    'SourcePredicateLearningFactInput',
    'SourcePredicateLearningFactResult',
    'SourceTripletFlashcardDraft',
    'SourceTripletFlashcardInput',
    'SourceTripletFlashcardResult',
    'SourceTripletLearningFact',
    'SourceTripletLearningFactEvidence',
    'SourceTripletLearningFactInput',
    'SourceTripletLearningFactResult',
]
