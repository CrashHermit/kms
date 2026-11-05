"""Image seam stage boundary models for KMS2."""

from pydantic import BaseModel, Field

from kms.core.model.content import MultimodalContent
from kms.core.model.page import SourcePage


class ImageSeamRequest(BaseModel):
    """One adjacent page pair presented to the image seam stage."""

    top_page: SourcePage
    bottom_page: SourcePage


class ImageSeamResult(BaseModel):
    """The image seam stage result for one adjacent page pair."""

    top_page: SourcePage
    bottom_page: SourcePage


class ImageSeamInput(BaseModel):
    """One isolated top/bottom pair of source material."""

    position: int = Field(ge=0, strict=True)
    top: MultimodalContent
    bottom: MultimodalContent


class ImageSeamDecision(BaseModel):
    """Continuation judgment for one packet-local visual seam."""

    position: int = Field(ge=0, strict=True)
    is_continuation: bool
