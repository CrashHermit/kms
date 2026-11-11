"""Image seam stage boundary models for KMS2."""

from pydantic import BaseModel, Field

from kms.media.models import Material
from kms.source.models.source import SourcePage


class ImageSeamRequest(BaseModel):
    """One adjacent page-pair candidate in the complete ordered candidate list."""

    position: int = Field(ge=0, strict=True)
    top_page: SourcePage
    bottom_page: SourcePage


class ImageSeamResult(BaseModel):
    """Continuation decision for one original adjacent page-pair candidate."""

    position: int = Field(ge=0, strict=True)
    is_continuation: bool


class ImageSeamInput(BaseModel):
    """One isolated top/bottom pair of source material."""

    position: int = Field(ge=0, strict=True)
    top: Material
    bottom: Material


class ImageSeamDecision(BaseModel):
    """Continuation judgment for one packet-local visual seam."""

    position: int = Field(ge=0, strict=True)
    is_continuation: bool
