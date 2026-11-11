"""Typed request and response models for statement reconstruction."""

from pydantic import BaseModel, Field

from kms.source.context import SourceBlockContext, project_source_block
from kms.source.models.material import StatementDraft
from kms.source.models.source import SourceBlock


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
    """A draft statement and its canonical reconstruction evidence."""

    target: StatementDraft
    target_blocks: list[SourceBlock]
    instruction_blocks: list[SourceBlock] = Field(default_factory=list)

    def model_input(self) -> SourceStatementReconstructionInput:
        """Project target and governing blocks as ordered native evidence."""
        return SourceStatementReconstructionInput(
            target_blocks=[
                project_source_block(block) for block in self.target_blocks
            ],
            instruction_blocks=[
                project_source_block(block) for block in self.instruction_blocks
            ],
        )
