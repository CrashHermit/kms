"""Content-only two-pass contracts for source-material flashcard generation.

Inference receives source evidence without backend UUIDs. Durable cards retain
their owning source-block provenance and selected parent-card provenance.
Regeneration clears generated cards and their review history for one source.
"""

from kms2.core.model.source_learning.atomic import (
    SourceAtomicFlashcardInput,
    SourceAtomicFlashcardRequest,
)
from kms2.core.model.source_learning.coherent import (
    SourceAtomicFlashcardPacket,
    SourceCoherentFlashcardCandidate,
    SourceCoherentFlashcardInput,
    SourceCoherentFlashcardPacketInput,
    SourceCoherentFlashcardRequest,
    SourceIndexedFlashcard,
)
from kms2.core.model.source_learning.context import SourceLearningEvidence
from kms2.core.model.source_learning.flashcard import (
    SourceFlashcard,
    SourceFlashcardCandidate,
    SourceFlashcardOccurrence,
)

__all__ = [
    'SourceAtomicFlashcardInput',
    'SourceAtomicFlashcardPacket',
    'SourceAtomicFlashcardRequest',
    'SourceCoherentFlashcardCandidate',
    'SourceCoherentFlashcardInput',
    'SourceCoherentFlashcardPacketInput',
    'SourceCoherentFlashcardRequest',
    'SourceFlashcard',
    'SourceFlashcardCandidate',
    'SourceFlashcardOccurrence',
    'SourceIndexedFlashcard',
    'SourceLearningEvidence',
]
