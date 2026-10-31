"""Typed request and response models for statement reconstruction."""

from pydantic import BaseModel, Field

from kms2.core.model.context import SourceBlockContext
from kms2.core.model.source_pedagogical.pedagogical import StatementDraft


class SourceStatementReconstructionInput(BaseModel):
    """Source evidence for one independently reconstructed statement."""

    target_blocks: list[SourceBlockContext]
    instruction_blocks: list[SourceBlockContext] = Field(default_factory=list)


class SourceStatementReconstructionItem(BaseModel):
    """One indexed statement reconstruction item."""

    position: int = Field(ge=0, strict=True)
    request: SourceStatementReconstructionInput


class SourceStatementReconstructionOutput(BaseModel):
    """One full reconstructed statement with its packet-local position."""

    position: int = Field(ge=0, strict=True)
    content: str


class SourceStatementReconstructionRequest(BaseModel):
    """A draft statement and its projected reconstruction evidence."""

    target: StatementDraft
    model_input: SourceStatementReconstructionInput
