"""Typed request and response models for procedure reconstruction."""

from pydantic import BaseModel, Field

from kms.source.context import SourceBlockContext, project_source_block
from kms.source.models.material import ProcedureDraft
from kms.source.models.source import SourceBlock


class SourceProcedureReconstructionInput(BaseModel):
    """Source evidence for one independently reconstructed procedure."""

    target_blocks: list[SourceBlockContext]


class SourceProcedureReconstructionItem(BaseModel):
    """One indexed procedure reconstruction item."""

    position: int = Field(ge=0, strict=True)
    request: SourceProcedureReconstructionInput


class SourceProcedureReconstructionOutput(BaseModel):
    """One full reconstructed procedure with its packet-local position."""

    position: int = Field(ge=0, strict=True)
    content: str


class SourceProcedureReconstructionRequest(BaseModel):
    """A draft procedure and its canonical reconstruction evidence."""

    target: ProcedureDraft
    target_blocks: list[SourceBlock]

    def model_input(self) -> SourceProcedureReconstructionInput:
        """Project target blocks as ordered native evidence."""
        return SourceProcedureReconstructionInput(
            target_blocks=[
                project_source_block(block) for block in self.target_blocks
            ]
        )
