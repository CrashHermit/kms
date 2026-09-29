"""Typed source-triplet learning contracts."""

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


class SourceTripletLearningFact(_LearningFactVertex):
    """Durable atomic learning fact about one source triplet hub."""


class SourceTripletLearningFactEvidence(_LearningFactEvidence):
    """Triplet learning evidence preserving the complete directed claim."""


class SourceTripletLearningFactInput(_LearningFactInput):
    """Triplet hub evidence supplied to the learning-fact model."""

    evidence: list[SourceTripletLearningFactEvidence]


class SourceTripletLearningFactResult(_LearningFactResult):
    """Atomic triplet learning facts returned by the model."""

    facts: list[_LearningFactOutput]


class SourceTripletFlashcardInput(_FlashcardInput):
    """One triplet learning fact supplied to the card model."""

    learning_fact: SourceTripletLearningFact
    evidence: list[SourceTripletLearningFactEvidence]


class SourceTripletFlashcardDraft(_FlashcardDraft):
    """One triplet card draft."""


class SourceTripletFlashcardResult(_FlashcardResult):
    """One validated triplet card result."""


__all__ = [
    'SourceTripletFlashcardDraft',
    'SourceTripletFlashcardInput',
    'SourceTripletFlashcardResult',
    'SourceTripletLearningFact',
    'SourceTripletLearningFactEvidence',
    'SourceTripletLearningFactInput',
    'SourceTripletLearningFactResult',
]
