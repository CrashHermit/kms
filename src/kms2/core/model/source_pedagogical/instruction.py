"""Instruction pointer model for KMS2."""

from pydantic import BaseModel, Field

from kms2.core.model.base import Vertex
from kms2.core.model.context import SourceBlockContext


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


class Instruction(Vertex):
    """An ordered block grouping with optional governed statements.

    Source ownership derives exclusively from nonempty, same-source block
    membership. Persistence retains MEMBER_OF and GOVERNS relationships,
    not a direct source ownership edge.
    """

    member_block_uuids: list[str] = Field(default_factory=list)
    governed_statement_uuids: list[str] = Field(default_factory=list)
