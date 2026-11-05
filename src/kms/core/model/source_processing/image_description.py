"""Image description stage boundary models for KMS2."""

from pydantic import BaseModel, Field

from kms.core.model.block import SourceBlock
from kms.core.model.content import MultimodalContent
from kms.core.model.context import SourceBlockContext, SourceBlockWindow


class ImageDescriptionRequest(BaseModel):
    """One image block and its surrounding source context."""

    flat_position: int
    source_block: SourceBlock
    window: SourceBlockWindow


class ImageDescriptionResult(BaseModel):
    """One image description with its flat source position."""

    flat_position: int
    description: str


class ImageDescriptionInput(BaseModel):
    """One logical image and its own context."""

    position: int = Field(ge=0, strict=True)
    target: MultimodalContent
    context_before: list[SourceBlockContext]
    context_after: list[SourceBlockContext]


class ImageDescriptionOutput(BaseModel):
    """Source-grounded description for one packet-local logical image."""

    position: int = Field(ge=0, strict=True)
    description: str
