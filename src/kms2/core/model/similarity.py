"""Typed projections returned by Neo4j vector searches."""

from pydantic import BaseModel

from kms2.core.model.block_types import BlockType


class SourceBlockSimilarityMatch(BaseModel):
    """Nearest source-block metadata without the indexed embedding."""

    uuid: str
    block_type: BlockType
    content: str | None
    score: float


class SourceEntitySimilarityMatch(BaseModel):
    """Nearest source-entity metadata without the indexed embedding."""

    uuid: str
    source_uuid: str
    source_block_uuid: str
    name: str
    description: str
    score: float


class SourceEventSimilarityMatch(BaseModel):
    """Nearest source-event metadata without the indexed embedding."""

    uuid: str
    source_uuid: str
    source_block_uuid: str
    name: str
    description: str
    score: float


class SourcePredicateSimilarityMatch(BaseModel):
    """Nearest source-predicate metadata without the indexed embedding."""

    uuid: str
    source_uuid: str
    source_block_uuid: str
    predicate: str
    description: str
    score: float


__all__ = [
    'SourceBlockSimilarityMatch',
    'SourceEntitySimilarityMatch',
    'SourceEventSimilarityMatch',
    'SourcePredicateSimilarityMatch',
]
