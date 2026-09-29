"""Typed source-predicate learning contracts."""

from kms2.core.model.source_learning._base import (
    _FlashcardDraft,
    _FlashcardInput,
    _FlashcardResult,
    _LearningFactEvidence,
    _LearningFactInput,
    _LearningFactOutput,
    _LearningFactResult,
    _LearningFactVertex,
)


class SourcePredicateLearningFact(_LearningFactVertex):
    """Durable atomic learning fact about one source predicate hub."""


class SourcePredicateLearningFactEvidence(_LearningFactEvidence):
    """Predicate learning evidence with complete endpoint context."""


class SourcePredicateLearningFactInput(_LearningFactInput):
    """Predicate hub evidence supplied to the learning-fact model."""

    evidence: list[SourcePredicateLearningFactEvidence]


class SourcePredicateLearningFactResult(_LearningFactResult):
    """Atomic predicate learning facts returned by the model."""

    facts: list[_LearningFactOutput]


class SourcePredicateFlashcardInput(_FlashcardInput):
    """One predicate learning fact supplied to the card model."""

    learning_fact: SourcePredicateLearningFact
    evidence: list[SourcePredicateLearningFactEvidence]


class SourcePredicateFlashcardDraft(_FlashcardDraft):
    """One predicate card draft."""


class SourcePredicateFlashcardResult(_FlashcardResult):
    """One validated predicate card result."""


__all__ = [
    'SourcePredicateFlashcardDraft',
    'SourcePredicateFlashcardInput',
    'SourcePredicateFlashcardResult',
    'SourcePredicateLearningFact',
    'SourcePredicateLearningFactEvidence',
    'SourcePredicateLearningFactInput',
    'SourcePredicateLearningFactResult',
]
