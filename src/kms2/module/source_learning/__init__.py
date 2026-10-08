"""DSPy modules for the two content-only source-learning passes."""

from kms2.module.source_learning.atomic import (
    SourceAtomicFlashcardModule,
    SourceAtomicFlashcardSignature,
)
from kms2.module.source_learning.coherent import (
    SourceCoherentFlashcardModule,
    SourceCoherentFlashcardSignature,
)

__all__ = [
    'SourceAtomicFlashcardModule',
    'SourceAtomicFlashcardSignature',
    'SourceCoherentFlashcardModule',
    'SourceCoherentFlashcardSignature',
]
