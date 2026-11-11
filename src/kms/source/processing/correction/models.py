"""Visual OCR content-correction transport models for KMS2."""

from pydantic import BaseModel, Field

from kms.media.models import ImageAsset, Material
from kms.source.models.source import BlockType, SourceBlock


class ContentCorrectionBlockContext(BaseModel):
    """Target block location in its request-local context image."""

    image: ImageAsset
    target_bounds: tuple[int, int, int, int]


class ContentCorrectionInput(BaseModel):
    """One locally positioned source block supplied for visual correction."""

    position: int = Field(strict=True, ge=0)
    block_type: BlockType
    material: Material
    context: ContentCorrectionBlockContext


class ContentCorrectionOutput(BaseModel):
    """Keep decision and corrected transcription for one local input position."""

    position: int = Field(strict=True, ge=0)
    keep: bool
    corrected_content: str | None
    cleanup_instructions: list[str]


class ContentCorrectionRequest(BaseModel):
    """One source block awaiting visual OCR correction."""

    page_index: int
    block_position: int
    source_block: SourceBlock
    context_image: bytes
    context_digest: str
    target_bounds: tuple[int, int, int, int]


class ContentCorrectionResult(BaseModel):
    """Corrected source block and cleanup decision at its page location."""

    page_index: int
    block_position: int
    source_block: SourceBlock
    keep: bool
    cleanup_instructions: list[str]
