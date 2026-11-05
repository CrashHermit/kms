"""Two-pass contracts for source-material flashcard generation.

Inference receives call-local source context without backend identities.
Durable cards retain their owning source-block and selected-parent provenance.
Regeneration clears generated cards and review history for one source.
"""

from kms.core.model.source_learning.atomic import (
    SourceAtomicFlashcardInput,
    SourceAtomicFlashcardMaterial,
    SourceAtomicFlashcardRequest,
)
from kms.core.model.source_learning.coherent import (
    SourceAtomicFlashcardPacket,
    SourceCoherentFlashcardCandidate,
    SourceCoherentFlashcardInput,
    SourceCoherentFlashcardPacketInput,
    SourceCoherentFlashcardRequest,
    SourceIndexedFlashcard,
)
from kms.core.model.source_learning.context import SourceLearningEvidence
from kms.core.model.source_learning.flashcard import (
    SourceFlashcard,
    SourceFlashcardCandidate,
    SourceFlashcardOccurrence,
)

__all__ = [
    'SourceAtomicFlashcardInput',
    'SourceAtomicFlashcardMaterial',
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
