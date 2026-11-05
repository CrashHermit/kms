"""Typed request and response models for statement reconstruction."""

from pydantic import BaseModel, Field

from kms.core.content import MultimodalContentBuilder
from kms.core.model.block import SourceBlock
from kms.core.model.context import SourceBlockContext
from kms.core.model.source_pedagogical.pedagogical import StatementDraft
from kms.core.source_context import project_source_block


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

    def model_input(
        self, builder: MultimodalContentBuilder
    ) -> SourceStatementReconstructionInput:
        """Project target and governing blocks into this call's catalog."""
        return SourceStatementReconstructionInput(
            target_blocks=[
                project_source_block(block, builder)
                for block in self.target_blocks
            ],
            instruction_blocks=[
                project_source_block(block, builder)
                for block in self.instruction_blocks
            ],
        )
