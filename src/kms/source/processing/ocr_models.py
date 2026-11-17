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


class OCRExtraction(BaseModel):
    """Flat OCR artifacts with independent physical-page capture metadata."""

    model_config = ConfigDict(extra='forbid')

    blocks: list[OCRArtifact] = Field(default_factory=list)
    page_indexes: list[int] = Field(default_factory=list)
    page_images: dict[int, str] = Field(default_factory=dict)
