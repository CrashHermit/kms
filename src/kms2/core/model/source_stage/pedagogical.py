"""Pedagogical discovery and pointer models for KMS2."""

from pydantic import BaseModel, Field

from kms2.core.model.base import Vertex
from kms2.core.model.context import SourceBlockContext


class PedagogicalComponent(BaseModel):
    """An in-memory ordered component discovered from source blocks."""

    member_block_uuids: list[str] = Field(default_factory=list)


class ExerciseComponent(BaseModel):
    """An in-memory ordered exercise component discovered from source blocks."""

    member_block_uuids: list[str] = Field(default_factory=list)


class StatementDraft(Vertex):
    """An ordered pointer to source blocks forming a statement."""

    member_block_uuids: list[str] = Field(default_factory=list)
    is_exercise: bool = False


class ProcedureDraft(Vertex):
    """An ordered pointer to source blocks forming a procedure."""

    member_block_uuids: list[str] = Field(default_factory=list)


class PedagogicalMember(BaseModel):
    """One zero-based local component member for role partitioning."""

    position: int
    source_block: SourceBlockContext
