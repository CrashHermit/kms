"""Typed source-event learning contracts."""

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


class SourceEventLearningFact(_LearningFactVertex):
    """Durable atomic learning fact about one source event hub."""


class SourceEventLearningFactEvidence(_LearningFactEvidence):
    """Event learning evidence with subject/object role provenance."""


class SourceEventLearningFactInput(_LearningFactInput):
    """Event hub evidence supplied to the learning-fact model."""

    evidence: list[SourceEventLearningFactEvidence]


class SourceEventLearningFactResult(_LearningFactResult):
    """Atomic event learning facts returned by the model."""

    facts: list[_LearningFactOutput]


class SourceEventFlashcardInput(_FlashcardInput):
    """One event learning fact supplied to the card model."""

    learning_fact: SourceEventLearningFact
    evidence: list[SourceEventLearningFactEvidence]


class SourceEventFlashcardDraft(_FlashcardDraft):
    """One event card draft."""


class SourceEventFlashcardResult(_FlashcardResult):
    """One validated event card result."""


__all__ = [
    'SourceEventFlashcardDraft',
    'SourceEventFlashcardInput',
    'SourceEventFlashcardResult',
    'SourceEventLearningFact',
    'SourceEventLearningFactEvidence',
    'SourceEventLearningFactInput',
    'SourceEventLearningFactResult',
]
