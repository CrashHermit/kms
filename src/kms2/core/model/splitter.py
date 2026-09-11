"""Source splitting boundary models for KMS2."""

from pydantic import BaseModel

from kms2.core.model.context import SourceBlockContext, SourceContextWindow


class SplitRequest(BaseModel):
    """One source-block split request with its surrounding context."""

    flat_position: int
    window: SourceContextWindow


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
