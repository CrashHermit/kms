"""Contextual image-filter stage boundary models."""

from pydantic import BaseModel, Field

from kms.core.model.block import SourceBlock
from kms.core.model.content import MultimodalContent
from kms.core.model.context import SourceBlockContext, SourceBlockWindow


class ImageFilterRequest(BaseModel):
    """One image target and its retained canonical source window."""

    flat_position: int
    source_block: SourceBlock
    window: SourceBlockWindow


class ImageFilterInput(BaseModel):
    """One logical image target and its own read-only source context."""

    position: int = Field(ge=0, strict=True)
    target: MultimodalContent
    context_before: list[SourceBlockContext]
    context_after: list[SourceBlockContext]


class ImageFilterDecision(BaseModel):
    """Keep or reject one packet-local image target."""

    position: int = Field(ge=0, strict=True)
    keep: bool


class ImageFilterResult(BaseModel):
    """One image decision restored to its flat source position."""

    flat_position: int
    keep: bool
