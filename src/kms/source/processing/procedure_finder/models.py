"""Typed classification contracts for procedure discovery."""

from pydantic import BaseModel, Field

from kms.source.context import SourceBlockContext


class ProcedureClassificationInput(BaseModel):
    position: int = Field(ge=0, strict=True)
    context_before: list[SourceBlockContext]
    target_block: SourceBlockContext
    context_after: list[SourceBlockContext]


class ProcedureStartDecision(BaseModel):
    position: int = Field(ge=0, strict=True)
    is_procedure_start: bool


class ProcedureBoundaryDecision(BaseModel):
    position: int = Field(ge=0, strict=True)
    is_procedure_boundary: bool
