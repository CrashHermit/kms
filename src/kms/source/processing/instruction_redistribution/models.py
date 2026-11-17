"""Transient instruction-support and applicability contracts."""

from pydantic import BaseModel, Field

from kms.source.context import SourceBlockContext


class ExerciseSupportGroup(BaseModel):
    block_uuids: list[str]


class InstructionClassificationInput(BaseModel):
    position: int = Field(ge=0, strict=True)
    context_before: list[SourceBlockContext]
    target_block: SourceBlockContext
    context_after: list[SourceBlockContext]


class InstructionStartDecision(BaseModel):
    position: int = Field(ge=0, strict=True)
    is_instruction_start: bool


class InstructionBoundaryDecision(BaseModel):
    position: int = Field(ge=0, strict=True)
    is_instruction_boundary: bool


class InstructionApplicabilityInput(BaseModel):
    position: int = Field(ge=0, strict=True)
    context_before: list[SourceBlockContext]
    exercise_blocks: list[SourceBlockContext]
    context_after: list[SourceBlockContext]


class InstructionApplicabilityDecision(BaseModel):
    position: int = Field(ge=0, strict=True)
    applies: bool
