"""Source-content formatting transport models for KMS2."""

from pydantic import BaseModel


class FormattingRequest(BaseModel):
    """One source block awaiting representation formatting."""

    page_index: int
    block_position: int
    content: str


class FormattingResult(BaseModel):
    """One source block's representation-formatting result."""

    page_index: int
    block_position: int
    formatted_content: str
