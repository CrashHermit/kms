"""Image seam stage boundary models for KMS2."""

from pydantic import BaseModel, Field

from kms.media.models import Material
from kms.source.models.source import SourceBlock


class ImageSeamRequest(BaseModel):
    """One adjacent physical-page candidate edge."""

    top_block: SourceBlock
    bottom_block: SourceBlock


class ImageSeamResult(BaseModel):
    """Continuation decision addressed to the original block pair."""

    top_uuid: str
    bottom_uuid: str
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
