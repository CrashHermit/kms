"""Source vertex models for KMS2."""

from typing import Any

from pydantic import BaseModel, Field

from kms2.core.model.base import Vertex
from kms2.core.model.block_types import BlockType


class VisualAsset(Vertex):
    """One visual asset attached to source content."""

    path: str


class Source(Vertex):
    """The root vertex that identifies one ingested source."""

    key: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceBlock(Vertex):
    """A canonical source block with content and visual assets."""

    block_type: BlockType
    content: str | None = None
    crop_path: str | None = None
    crop_bbox: tuple[int, int, int, int] | None = None
    assets: list[VisualAsset] = Field(default_factory=list)


class SourcePage(BaseModel):
    """An ordered source page containing canonical source blocks."""

    index: int
    blocks: list[SourceBlock] = Field(default_factory=list)
