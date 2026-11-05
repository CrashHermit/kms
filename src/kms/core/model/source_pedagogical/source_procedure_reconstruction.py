"""Typed request and response models for procedure reconstruction."""

from pydantic import BaseModel, Field

from kms.core.content import MultimodalContentBuilder
from kms.core.model.block import SourceBlock
from kms.core.model.context import SourceBlockContext
from kms.core.model.source_pedagogical.pedagogical import ProcedureDraft
from kms.core.source_context import project_source_block


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

    def model_input(
        self, builder: MultimodalContentBuilder
    ) -> SourceProcedureReconstructionInput:
        """Project target blocks into this call's image catalog."""
        return SourceProcedureReconstructionInput(
            target_blocks=[
                project_source_block(block, builder)
                for block in self.target_blocks
            ]
        )
