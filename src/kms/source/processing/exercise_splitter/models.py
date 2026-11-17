"""Source exercise-splitting boundary models."""

from pydantic import BaseModel, Field

from kms.source.context import SourceBlockContext, SourceBlockWindow


class SplitRequest(BaseModel):
    """One source-block split request with its surrounding context."""

    flat_position: int
    window: SourceBlockWindow


class SplitRouteDecision(BaseModel):
    """Whether a target block contains exercise entry/exit boundaries."""

    position: int = Field(ge=0, strict=True)
    requires_split: bool


class SplitCandidate(BaseModel):
    """One target-local source block selected for splitting."""

    position: int
    source_block: SourceBlockContext


class SplitPiece(BaseModel):
    """One verbatim replacement fragment of a source block."""

    content: str


class SplitResult(BaseModel):
    """One dispatched source-block split result."""

    flat_position: int
    pieces: list[SplitPiece] | None = None


class SplitDecision(BaseModel):
    """The ordered replacement pieces for one split candidate."""

    position: int
    pieces: list[SplitPiece]
