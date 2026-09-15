"""Image description stage boundary models for KMS2."""

from pydantic import BaseModel

from kms2.core.model.block import SourceBlock
from kms2.core.model.context import SourceContextWindow


class ImageDescriptionRequest(BaseModel):
    """One image block and its surrounding source context."""

    flat_position: int
    source_block: SourceBlock
    window: SourceContextWindow


class ImageDescriptionResult(BaseModel):
    """One image description with its flat source position."""

    flat_position: int
    description: str
