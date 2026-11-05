"""Source splitting boundary models for KMS2."""

from pydantic import BaseModel, Field

from kms.core.model.context import SourceBlockContext, SourceBlockWindow


class SplitRequest(BaseModel):
    """One source-block split request with its surrounding context."""

    flat_position: int
    window: SourceBlockWindow


class SplitRouteDecision(BaseModel):
    """Whether one target-local source block contains multiple exercises."""

    position: int = Field(ge=0, strict=True)
    contains_multiple_exercises: bool


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
