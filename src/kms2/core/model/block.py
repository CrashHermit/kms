"""Canonical source block vertex model."""

from pydantic import Field

from kms2.core.model.base import Vertex
from kms2.core.model.block_types import BlockType
from kms2.core.model.visual_asset import VisualAsset


class SourceBlock(Vertex):
    """A canonical source block with content and visual assets."""

    block_type: BlockType
    content: str | None = None
    embedding: list[float] | None = None
    crop_path: str | None = None
    crop_bbox: tuple[int, int, int, int] | None = None
    assets: list[VisualAsset] = Field(default_factory=list)


__all__ = ['SourceBlock']
