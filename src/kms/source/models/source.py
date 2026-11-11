"""Canonical source-domain models."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from kms.core.models import Vertex


class Source(Vertex):
    """The root vertex that identifies one ingested source."""

    key: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    revision: int = Field(default=0, ge=0)


class BlockType(StrEnum):
    """Structural types assigned to canonical source blocks."""

    PARAGRAPH = 'paragraph'
    EQUATION = 'equation'
    MATH = 'math'
    CODE = 'code'
    LIST = 'list'
    TABLE = 'table'
    IMAGE = 'image'
    CAPTION = 'caption'
    HEADER = 'header'
    BIBLIOGRAPHIC = 'bibliographic'
    NOTE = 'note'
    FOOTER = 'footer'
    ASIDE_TEXT = 'aside_text'
    MARKDOWN = 'markdown'
    INSTRUCTION = 'instruction'


class SourceBlock(Vertex):
    """A canonical source block with optional directly owned image data."""

    block_type: BlockType
    content: str | None = None
    origin_uuid: str | None = None
    embedding: list[float] | None = None
    crop_path: str | None = None
    crop_sha256: str | None = None
    crop_bbox: tuple[int, int, int, int] | None = None
    image_path: str | None = None
    image_sha256: str | None = None
    image_page_indexes: list[int] = Field(default_factory=list)

    @model_validator(mode='after')
    def validate_image_provenance(self) -> SourceBlock:
        """Require every image block to identify its image and source pages."""
        if self.block_type is BlockType.IMAGE:
            if not self.image_path or not self.image_page_indexes:
                raise ValueError(
                    'Image blocks require an image path and page provenance'
                )
        elif (
            self.image_path is not None
            or self.image_sha256 is not None
            or self.image_page_indexes
        ):
            raise ValueError('Non-image blocks cannot own image data')
        return self


class SourcePage(BaseModel):
    """An ordered source page containing markdown and canonical blocks."""

    index: int
    markdown: str = ''
    blocks: list[SourceBlock] = Field(default_factory=list)
