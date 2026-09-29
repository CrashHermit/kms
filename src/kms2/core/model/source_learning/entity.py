"""Typed source-entity learning contracts."""

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


class SourceEntityLearningFact(_LearningFactVertex):
    """Durable atomic learning fact about one source entity hub."""


class SourceEntityLearningFactEvidence(_LearningFactEvidence):
    """Entity learning evidence with subject/object role provenance."""


class SourceEntityLearningFactInput(_LearningFactInput):
    """Entity hub evidence supplied to the learning-fact model."""

    evidence: list[SourceEntityLearningFactEvidence]


class SourceEntityLearningFactResult(_LearningFactResult):
    """Atomic entity learning facts returned by the model."""

    facts: list[_LearningFactOutput]


class SourceEntityFlashcardInput(_FlashcardInput):
    """One entity learning fact supplied to the card model."""

    learning_fact: SourceEntityLearningFact
    evidence: list[SourceEntityLearningFactEvidence]


class SourceEntityFlashcardDraft(_FlashcardDraft):
    """One entity card draft."""


class SourceEntityFlashcardResult(_FlashcardResult):
    """One validated entity card result."""


__all__ = [
    'SourceEntityFlashcardDraft',
    'SourceEntityFlashcardInput',
    'SourceEntityFlashcardResult',
    'SourceEntityLearningFact',
    'SourceEntityLearningFactEvidence',
    'SourceEntityLearningFactInput',
    'SourceEntityLearningFactResult',
]
