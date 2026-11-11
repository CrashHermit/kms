"""Typed contracts for instruction discovery."""

from pydantic import BaseModel, Field

from kms.source.context import SourceBlockContext


class InstructionClassificationInput(BaseModel):
    """One independently contextualized instruction-classification target."""

    position: int = Field(ge=0, strict=True)
    context_before: list[SourceBlockContext]
    target_block: SourceBlockContext
    context_after: list[SourceBlockContext]


class InstructionStartDecision(BaseModel):
    """Classification result for one local instruction-start position."""

    position: int = Field(ge=0, strict=True)
    is_instruction_start: bool


class InstructionBoundaryDecision(BaseModel):
    """Classification result for one local instruction-boundary position."""

    position: int = Field(ge=0, strict=True)
    is_instruction_boundary: bool
