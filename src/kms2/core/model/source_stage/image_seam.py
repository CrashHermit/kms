"""Image seam stage boundary models for KMS2."""

from pydantic import BaseModel

from kms2.core.model.page import SourcePage


class ImageSeamRequest(BaseModel):
    """One adjacent page pair presented to the image seam stage."""

    top_page: SourcePage
    bottom_page: SourcePage


class ImageSeamResult(BaseModel):
    """The image seam stage result for one adjacent page pair."""

    top_page: SourcePage
    bottom_page: SourcePage
