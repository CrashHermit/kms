"""Pedagogical discovery and pointer models for KMS2."""

from pydantic import BaseModel, Field

from kms2.core.model.base import Vertex
from kms2.core.model.context import SourceBlockContext


class PedagogicalClassificationInput(BaseModel):
    """One independently contextualized pedagogical-classification target."""

    position: int = Field(ge=0, strict=True)
    context_before: list[SourceBlockContext]
    target_block: SourceBlockContext
    context_after: list[SourceBlockContext]


class PedagogicalStartDecision(BaseModel):
    """Classification result for one local pedagogical-start position."""

    position: int = Field(ge=0, strict=True)
    is_pedagogical_start: bool


class PedagogicalBoundaryDecision(BaseModel):
    """Classification result for one local pedagogical-boundary position."""

    position: int = Field(ge=0, strict=True)
    is_pedagogical_boundary: bool


class ExerciseClassificationInput(BaseModel):
    """One independently contextualized exercise-classification target."""

    position: int = Field(ge=0, strict=True)
    context_before: list[SourceBlockContext]
    target_block: SourceBlockContext
    context_after: list[SourceBlockContext]


class ExerciseStartDecision(BaseModel):
    """Classification result for one local exercise-start position."""

    position: int = Field(ge=0, strict=True)
    is_exercise_start: bool


class ExerciseBoundaryDecision(BaseModel):
    """Classification result for one local exercise-boundary position."""

    position: int = Field(ge=0, strict=True)
    is_exercise_boundary: bool


class PedagogicalComponent(BaseModel):
    """An in-memory ordered component discovered from source blocks."""

    member_block_uuids: list[str] = Field(default_factory=list)


class ExerciseComponent(BaseModel):
    """An in-memory ordered exercise component discovered from source blocks."""

    member_block_uuids: list[str] = Field(default_factory=list)


class StatementDraft(Vertex):
    """An ordered pointer to source blocks forming a statement."""

    content: str = ''
    embedding: list[float] | None = None
    member_block_uuids: list[str] = Field(default_factory=list)
    is_exercise: bool = False


class ProcedureDraft(Vertex):
    """An ordered pointer to source blocks forming a procedure."""

    content: str = ''
    embedding: list[float] | None = None
    member_block_uuids: list[str] = Field(default_factory=list)


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
