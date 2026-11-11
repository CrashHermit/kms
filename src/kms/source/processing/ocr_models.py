"""OCR adapter transport models for KMS2."""

from pydantic import BaseModel, ConfigDict, Field

from kms.source.models.source import BlockType


class OCRArtifact(BaseModel):
    """One OCR block and its correction crop provenance."""

    model_config = ConfigDict(extra='forbid')

    page_index: int
    block_index: int
    block_type: BlockType
    content: str | None = None
    crop_path: str | None = None
    crop_bbox: tuple[int, int, int, int] | None = None


class OCRPageArtifact(BaseModel):
    """One OCR page, its rendered image path, and ordered block artifacts."""

    model_config = ConfigDict(extra='forbid')

    page_index: int
    page_image_path: str
    markdown: str
    blocks: list[OCRArtifact] = Field(default_factory=list)
