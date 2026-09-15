"""OCR adapter transport models for KMS2."""

from pydantic import BaseModel, ConfigDict, Field

from kms2.core.model.block_types import BlockType


class OCRImageArtifact(BaseModel):
    """One materialized OCR image and its source-page geometry.

    ``bbox`` is an optional ``(left, top, right, bottom)`` fraction of the
    source page. The file at ``path`` is written by the OCR provider adapter.
    """

    model_config = ConfigDict(extra='forbid')

    path: str
    image_id: str | None = None
    bbox: tuple[float, float, float, float] | None = None


class OCRArtifact(BaseModel):
    """One OCR block and its visual correction artifacts.

    ``crop_bbox`` is an optional ``(left, top, right, bottom)`` box in pixels
    of the rendered source page image. ``images`` contains provider images
    whose page-space boxes overlap this block.
    """

    model_config = ConfigDict(extra='forbid')

    page_index: int
    block_index: int
    block_type: BlockType
    content: str | None = None
    images: list[OCRImageArtifact] = Field(default_factory=list)
    crop_path: str | None = None
    crop_bbox: tuple[int, int, int, int] | None = None


class OCRPageArtifact(BaseModel):
    """One OCR page with its markdown and ordered block artifacts."""

    model_config = ConfigDict(extra='forbid')

    page_index: int
    markdown: str
    blocks: list[OCRArtifact] = Field(default_factory=list)
