"""Source splitting boundary models for KMS2."""

from pydantic import BaseModel

from kms2.core.model.context import SourceBlockContext


class SplitCandidate(BaseModel):
    """One target-local source block selected for splitting."""

    position: int
    source_block: SourceBlockContext


class SplitPiece(BaseModel):
    """One verbatim replacement fragment of a source block."""

    content: str


class SplitDecision(BaseModel):
    """The ordered replacement pieces for one split candidate."""

    position: int
    pieces: list[SplitPiece]
