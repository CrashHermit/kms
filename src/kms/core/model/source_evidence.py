"""Transient source-search and evidence-reading projections."""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from kms.core.model.block import SourceBlock
from kms.core.model.source import Source
from kms.core.model.source_semantic.source_procedure import SourceProcedure
from kms.core.model.source_semantic.source_statement import SourceStatement
from kms.core.model.teaching.positions import Position
from kms.core.model.visual_asset import VisualAsset


class SourceEvidenceBlock(BaseModel):
    """One source block with its containing page locations."""

    block: SourceBlock
    page_indexes: list[int]


class SourceSearchCandidate(SourceEvidenceBlock):
    """An eligible source block scored by embedding similarity."""

    source: Source
    similarity: float


class SourceSearchHit(SourceSearchCandidate):
    """A search candidate with its final reranker score."""

    relevance_score: float


class SourceStatementSearchCandidate(BaseModel):
    """An owned reconstructed statement scored by embedding similarity."""

    statement: SourceStatement
    source: Source
    page_indexes: list[int]
    similarity: float


class SourceStatementSearchHit(SourceStatementSearchCandidate):
    """A statement candidate with its final reranker score."""

    relevance_score: float


class SourceProcedureSearchCandidate(BaseModel):
    """An owned reconstructed procedure scored by embedding similarity."""

    procedure: SourceProcedure
    source: Source
    page_indexes: list[int]
    similarity: float


class SourceProcedureSearchHit(SourceProcedureSearchCandidate):
    """A procedure candidate with its final reranker score."""

    relevance_score: float


class SourceEvidenceGroup(BaseModel):
    """A direct statement, procedure, or instruction membership boundary."""

    uuid: str
    kind: Literal['statement', 'procedure', 'instruction']
    member_block_uuids: list[str]
    governed_statement_uuids: list[str] = Field(default_factory=list)


class SourcePassage(BaseModel):
    """Selected blocks and directly related pedagogical group boundaries."""

    source: Source
    selected_block_uuids: list[str]
    blocks: list[SourceEvidenceBlock]
    groups: list[SourceEvidenceGroup]


class SourceStatementPassage(BaseModel):
    """A reconstructed statement with its complete source passages."""

    statement: SourceStatement
    passages: list[SourcePassage]


class SourceProcedurePassage(BaseModel):
    """A reconstructed procedure with its complete source passages."""

    procedure: SourceProcedure
    passages: list[SourcePassage]


@dataclass(frozen=True, slots=True)
class SourceImage:
    """Original source image bytes authorized through a source graph."""

    asset: VisualAsset
    data: bytes


class SourceReadResult(BaseModel):
    """Material read into canonical workspace registries without inspection."""

    passages: list[SourcePassage] = Field(default_factory=list)
    block_positions: list[Position] = Field(default_factory=list)
    image_positions: list[Position] = Field(default_factory=list)
