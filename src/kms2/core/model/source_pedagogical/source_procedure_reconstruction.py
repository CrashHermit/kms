"""Typed request and response models for procedure reconstruction."""

from pydantic import BaseModel, Field

from kms2.core.model.context import SourceBlockContext
from kms2.core.model.source_pedagogical.pedagogical import ProcedureDraft


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
    """A draft procedure and its projected reconstruction evidence."""

    target: ProcedureDraft
    model_input: SourceProcedureReconstructionInput
