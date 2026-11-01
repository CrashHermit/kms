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
    target_block_uuids: list[str]
    name: str
    description: str
    score: float


class SourceEntityHubSearchMatch(BaseModel):
    """Best source-local entity occurrence matched through its hub."""

    uuid: str
    source_uuid: str
    member_uuid: str
    target_block_uuids: list[str]
    name: str
    description: str
    score: float
