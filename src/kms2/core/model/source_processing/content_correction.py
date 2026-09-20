"""Visual OCR content-correction transport models for KMS2."""

from pydantic import BaseModel

from kms2.core.model.block import SourceBlock


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
