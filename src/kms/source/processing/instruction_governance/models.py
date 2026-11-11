"""Typed contracts for instruction governance."""

from pydantic import BaseModel, Field

from kms.source.context import SourceBlockContext


class InstructionGovernanceInput(BaseModel):
    """One complete statement and its independent reference context."""

    position: int = Field(ge=0, strict=True)
    context_before: list[SourceBlockContext]
    statement_blocks: list[SourceBlockContext]
    context_after: list[SourceBlockContext]


class InstructionGovernanceDecision(BaseModel):
    """Governance judgment for a statement under the shared instruction."""

    position: int = Field(ge=0, strict=True)
    governs: bool
