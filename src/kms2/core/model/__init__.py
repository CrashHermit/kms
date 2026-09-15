"""KMS2 graph and source-content models."""

from kms2.core.model.base import Edge, Vertex
from kms2.core.model.block import SourceBlock
from kms2.core.model.block_types import BlockType
from kms2.core.model.context import (
    SourceBlockContext,
    SourceContextWindow,
)
from kms2.core.model.page import SourcePage
from kms2.core.model.semantic import (
    AtomicFact,
    ExtractedFact,
    FactExtractionInput,
    FactExtractionRequest,
    FactExtractionResult,
    RawAssertion,
    RawTriplet,
    SemanticNodeKind,
    SourceEntity,
    SourceEntityDescriptionRequest,
    SourceEntityDescriptionResult,
    SourceEntityDescriptionTarget,
    SourceEvent,
    SourceEventDescriptionRequest,
    SourceEventDescriptionResult,
    SourceEventDescriptionTarget,
    SourcePredicate,
    SourcePredicateDescriptionRequest,
    SourcePredicateDescriptionResult,
    SourcePredicateDescriptionTarget,
    TermDescriptionInput,
    TripletCandidate,
    TripletDecompositionRequest,
    TripletDecompositionResult,
)
from kms2.core.model.similarity import (
    SourceBlockSimilarityMatch,
    SourceEntitySimilarityMatch,
    SourceEventSimilarityMatch,
    SourcePredicateSimilarityMatch,
)
from kms2.core.model.source import Source
from kms2.core.model.source_stage.content_correction import (
    ContentCorrectionRequest,
    ContentCorrectionResult,
)
from kms2.core.model.source_stage.embedding import (
    EmbeddingRequest,
    EmbeddingResult,
)
from kms2.core.model.source_stage.formatting import (
    FormattingRequest,
    FormattingResult,
)
from kms2.core.model.source_stage.image_description import (
    ImageDescriptionRequest,
    ImageDescriptionResult,
)
from kms2.core.model.source_stage.image_seam import (
    ImageSeamRequest,
    ImageSeamResult,
)
from kms2.core.model.source_stage.instruction import Instruction
from kms2.core.model.source_stage.ocr import (
    OCRArtifact,
    OCRImageArtifact,
    OCRPageArtifact,
)
from kms2.core.model.source_stage.pedagogical import (
    ExerciseComponent,
    PedagogicalComponent,
    PedagogicalMember,
    Procedure,
    Statement,
)
from kms2.core.model.source_stage.splitter import (
    SplitCandidate,
    SplitDecision,
    SplitPiece,
    SplitRequest,
    SplitResult,
)
from kms2.core.model.source_stage.text_seam import (
    TextSeamRequest,
    TextSeamResult,
)
from kms2.core.model.visual_asset import VisualAsset

__all__ = [
    'EmbeddingRequest',
    'EmbeddingResult',
    'BlockType',
    'ContentCorrectionRequest',
    'ContentCorrectionResult',
    'Edge',
    'ExerciseComponent',
    'Instruction',
    'PedagogicalComponent',
    'PedagogicalMember',
    'Procedure',
    'Statement',
    'FormattingRequest',
    'FormattingResult',
    'ImageDescriptionRequest',
    'ImageDescriptionResult',
    'ImageSeamRequest',
    'ImageSeamResult',
    'OCRArtifact',
    'OCRImageArtifact',
    'OCRPageArtifact',
    'Source',
    'SourceBlock',
    'SourceBlockContext',
    'SourceContextWindow',
    'SourcePage',
    'TextSeamRequest',
    'SplitCandidate',
    'SplitDecision',
    'SplitPiece',
    'SplitRequest',
    'SplitResult',
    'TextSeamResult',
    'Vertex',
    'VisualAsset',
    'AtomicFact',
    'SourceEntity',
    'SourceEntityDescriptionRequest',
    'SourceEntityDescriptionResult',
    'SourceEntityDescriptionTarget',
    'SourceEvent',
    'SourceEventDescriptionRequest',
    'SourceEventDescriptionResult',
    'SourceEventDescriptionTarget',
    'ExtractedFact',
    'FactExtractionInput',
    'FactExtractionRequest',
    'FactExtractionResult',
    'SourcePredicate',
    'SourcePredicateDescriptionRequest',
    'SourcePredicateDescriptionResult',
    'SourcePredicateDescriptionTarget',
    'RawAssertion',
    'RawTriplet',
    'SemanticNodeKind',
    'TermDescriptionInput',
    'TripletCandidate',
    'TripletDecompositionRequest',
    'TripletDecompositionResult',
    'SourceBlockSimilarityMatch',
    'SourceEntitySimilarityMatch',
    'SourceEventSimilarityMatch',
    'SourcePredicateSimilarityMatch',
]
