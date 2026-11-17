"""Text seam transport models for KMS2."""

from pydantic import BaseModel, Field

from kms.source.context import SourceBlockContext
from kms.source.models.source import SourceBlock


class TextSeamRequest(BaseModel):
    """One original physical-boundary candidate."""

    tail: SourceBlock
    head: SourceBlock
    top_context: SourceBlock | None = None
    bottom_context: SourceBlock | None = None


class TextSeamResult(BaseModel):
    """Split decision addressed to its original block pair."""

    tail_uuid: str
    head_uuid: str
    is_split: bool


class TextSeamRewriteRequest(BaseModel):
    """One accepted maximal chain awaiting rewrite."""

    blocks: list[SourceBlock]
    top_context: SourceBlock | None = None
    bottom_context: SourceBlock | None = None


class TextSeamRewriteInput(BaseModel):
    """One request-local, ordered interrupted-block chain."""

    position: int = Field(ge=0, strict=True)
    blocks: list[SourceBlockContext]
    top_context: SourceBlockContext | None = None
    bottom_context: SourceBlockContext | None = None


class TextSeamInput(BaseModel):
    """One independent adjacent-block seam judgment."""

    position: int = Field(ge=0, strict=True)
    tail: SourceBlockContext
    head: SourceBlockContext
    top_context: SourceBlockContext | None
    bottom_context: SourceBlockContext | None


class TextSeamDecision(BaseModel):
    """Split judgment for one packet-local seam position."""

    position: int = Field(ge=0, strict=True)
    is_split: bool


class TextSeamRewriteOutput(BaseModel):
    """Rejoined text retaining its packet-local chain position."""

    position: int = Field(ge=0, strict=True)
    merged: str


class TextSeamRewriteResult(BaseModel):
    """Merged chain content addressed by its original block UUIDs."""

    block_uuids: list[str]
    merged: str
