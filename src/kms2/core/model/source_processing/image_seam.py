"""Image seam stage boundary models for KMS2."""

from pydantic import BaseModel, Field

from kms2.core.model.page import SourcePage


class ImageSeamRequest(BaseModel):
    """One adjacent page pair presented to the image seam stage."""

    top_page: SourcePage
    bottom_page: SourcePage


class ImageSeamResult(BaseModel):
    """The image seam stage result for one adjacent page pair."""

    top_page: SourcePage
    bottom_page: SourcePage


class ImageSeamInput(BaseModel):
    """One pair of native-image references within a seam packet."""

    position: int = Field(ge=0, strict=True)
    top_image_position: int = Field(ge=0, strict=True)
    bottom_image_position: int = Field(ge=0, strict=True)


class ImageSeamDecision(BaseModel):
    """Continuation judgment for one packet-local visual seam."""

    position: int = Field(ge=0, strict=True)
    is_continuation: bool
