"""Text seam transport models for KMS2."""

from pydantic import BaseModel

from kms2.core.model.source import SourcePage


class TextSeamRequest(BaseModel):
    """One adjacent source-page pair awaiting seam judgment."""

    top_page: SourcePage
    bottom_page: SourcePage


class TextSeamResult(BaseModel):
    """One adjacent source-page pair after seam processing."""

    top_page: SourcePage
    bottom_page: SourcePage
