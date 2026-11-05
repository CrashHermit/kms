"""Canonical source block vertex model."""

from pydantic import Field

from kms.core.model.base import Vertex
from kms.core.model.block_types import BlockType
from kms.core.model.visual_asset import VisualAsset


class SourceBlock(Vertex):
    """A canonical source block with content and visual assets."""

    block_type: BlockType
    content: str | None = None
    origin_uuid: str | None = None
    embedding: list[float] | None = None
    crop_path: str | None = None
    crop_sha256: str | None = None
    crop_bbox: tuple[int, int, int, int] | None = None
    assets: list[VisualAsset] = Field(default_factory=list)
