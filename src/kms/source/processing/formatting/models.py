"""Source-content formatting transport models for KMS2."""

from typing import Annotated

from pydantic import BaseModel, Field, StrictInt


class FormattingInput(BaseModel):
    """One independently formatted block with a batch-local position."""

    position: Annotated[StrictInt, Field(ge=0)]
    content: str
    cleanup_instructions: list[str] = Field(default_factory=list)


class FormattingOutput(BaseModel):
    """One formatted block identified by its batch-local position."""

    position: Annotated[StrictInt, Field(ge=0)]
    formatted_content: str


class FormattingRequest(BaseModel):
    """One source block awaiting representation formatting."""

    block_uuid: str
    content: str
    cleanup_instructions: list[str] = Field(default_factory=list)


class FormattingResult(BaseModel):
    """One source block's representation-formatting result."""

    block_uuid: str
    formatted_content: str
