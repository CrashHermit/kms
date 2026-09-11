"""Image enrichment stage boundary models for KMS2."""

from pydantic import BaseModel

from kms2.core.model.context import SourceContextWindow
from kms2.core.model.source import SourceBlock


class ImageEnrichmentRequest(BaseModel):
    """One image block and its surrounding source context."""

    flat_position: int
    source_block: SourceBlock
    window: SourceContextWindow


class ImageEnrichmentResult(BaseModel):
    """One enriched image description with its flat source position."""

    flat_position: int
    description: str
