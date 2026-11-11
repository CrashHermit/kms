"""Text seam transport models for KMS2."""

from pydantic import BaseModel, Field

from kms.source.context import SourceBlockContext
from kms.source.models.source import SourcePage


class TextSeamRequest(BaseModel):
    """One adjacent source-page pair awaiting seam judgment."""

    top_page: SourcePage
    bottom_page: SourcePage


class TextSeamResult(BaseModel):
    """One adjacent source-page pair after seam processing."""

    top_page: SourcePage
    bottom_page: SourcePage


class TextSeamInput(BaseModel):
    """One interrupted-block candidate with reference-only structural neighbors."""

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
    """Rejoined text retaining the original judge-packet position."""

    position: int = Field(ge=0, strict=True)
    merged: str
