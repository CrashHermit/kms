"""Typed DSPy modules for source learning."""

from kms2.module.source_learning.entity import (
    SourceEntityFlashcardModule,
    SourceEntityFlashcardSignature,
    SourceEntityLearningFactModule,
    SourceEntityLearningFactSignature,
)
from kms2.module.source_learning.event import (
    SourceEventFlashcardModule,
    SourceEventFlashcardSignature,
    SourceEventLearningFactModule,
    SourceEventLearningFactSignature,
)
from kms2.module.source_learning.predicate import (
    SourcePredicateFlashcardModule,
    SourcePredicateFlashcardSignature,
    SourcePredicateLearningFactModule,
    SourcePredicateLearningFactSignature,
)
from kms2.module.source_learning.triplet import (
    SourceTripletFlashcardModule,
    SourceTripletFlashcardSignature,
    SourceTripletLearningFactModule,
    SourceTripletLearningFactSignature,
)

__all__ = [
    'SourceEntityFlashcardModule',
    'SourceEntityFlashcardSignature',
    'SourceEntityLearningFactModule',
    'SourceEntityLearningFactSignature',
    'SourceEventFlashcardModule',
    'SourceEventFlashcardSignature',
    'SourceEventLearningFactModule',
    'SourceEventLearningFactSignature',
    'SourcePredicateFlashcardModule',
    'SourcePredicateFlashcardSignature',
    'SourcePredicateLearningFactModule',
    'SourcePredicateLearningFactSignature',
    'SourceTripletFlashcardModule',
    'SourceTripletFlashcardSignature',
    'SourceTripletLearningFactModule',
    'SourceTripletLearningFactSignature',
]
