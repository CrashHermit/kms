"""Typed contracts for pedagogical material partitioning."""

from pydantic import BaseModel, Field

from kms.source.context import SourceBlockContext


class PedagogicalMember(BaseModel):
    """One zero-based local component member for role partitioning."""

    position: int
    source_block: SourceBlockContext


class PedagogicalComponentInput(BaseModel):
    """One complete component with independently local member positions."""

    position: int = Field(ge=0, strict=True)
    members: list[PedagogicalMember]


class PedagogicalRoleDecision(BaseModel):
    """Statement and working roles for one packet-local component."""

    position: int = Field(ge=0, strict=True)
    has_statement: bool
    has_procedure: bool


class StatementPartitionDecision(BaseModel):
    """Component-local member positions belonging to its statement."""

    position: int = Field(ge=0, strict=True)
    statement_positions: list[int]


class ProcedurePartitionDecision(BaseModel):
    """Component-local member positions belonging to its working."""

    position: int = Field(ge=0, strict=True)
    procedure_positions: list[int]
