"""Content-only two-pass contracts for source-local flashcard generation.

Atomic and coherent inference receives semantic content without backend UUIDs.
Transient packets retain provenance for collection; every durable result is a
generic source flashcard with direct triplet, fact, hub, and parent-card edges.
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
from kms2.core.model.source_learning.context import (
    SourceLearningEvidence,
    SourceLearningHubContext,
)
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
    'SourceLearningHubContext',
]
