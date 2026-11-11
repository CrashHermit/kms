"""Source retrieval and evidence contracts."""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from kms.source.models.material import SourceProcedure, SourceStatement
from kms.source.models.source import Source, SourceBlock


class SourceImageReference(BaseModel):
    """Authorized path and digest for one owned source image block."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    block_uuid: str
    image_path: str

    image_sha256: str


class SourceEvidenceBlock(BaseModel):
    """One source block with its containing page locations."""

    block: SourceBlock
    page_indexes: list[int]


class SourceSearchHit(SourceEvidenceBlock):
    """An owned source block with embedding similarity and page provenance."""

    source: Source
    similarity: float


class SourceStatementSearchHit(BaseModel):
    """An owned statement with embedding similarity and page provenance."""

    statement: SourceStatement
    source: Source
    page_indexes: list[int]
    similarity: float


class SourceProcedureSearchHit(BaseModel):
    """An owned procedure with embedding similarity and page provenance."""

    procedure: SourceProcedure
    source: Source
    page_indexes: list[int]
    similarity: float


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
    """Canonical image bytes authorized through an owning source block."""

    block_uuid: str
    data: bytes
