"""Visual OCR content-correction transport models for KMS2."""

from pydantic import BaseModel, Field

from kms2.core.model.block import SourceBlock
from kms2.core.model.block_types import BlockType


class ContentCorrectionInput(BaseModel):
    """One locally positioned source block supplied for visual correction."""

    position: int = Field(strict=True, ge=0)
    block_type: BlockType
    content: str | None


class ContentCorrectionOutput(BaseModel):
    """Corrected transcription identified by its local input position."""

    position: int = Field(strict=True, ge=0)
    corrected_content: str | None


class ContentCorrectionRequest(BaseModel):
    """One source block awaiting visual OCR correction."""

    page_index: int
    block_position: int
    source_block: SourceBlock


class ContentCorrectionResult(BaseModel):
    """One corrected source block with its page location."""

    page_index: int
    block_position: int
    source_block: SourceBlock
