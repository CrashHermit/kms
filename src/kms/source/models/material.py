"""Source-owned reconstruction and pedagogical material records."""

from pydantic import BaseModel, Field

from kms.core.models import Vertex


class Instruction(Vertex):
    """An ordered block grouping with optional governed statements.

    Source ownership derives exclusively from nonempty, same-source block
    membership. Persistence retains MEMBER_OF and GOVERNS relationships,
    not a direct source ownership edge.
    """

    member_block_uuids: list[str] = Field(default_factory=list)
    governed_statement_uuids: list[str] = Field(default_factory=list)


class ExerciseComponent(BaseModel):
    """An in-memory ordered exercise component discovered from source blocks."""

    member_block_uuids: list[str] = Field(default_factory=list)


class PedagogicalComponent(BaseModel):
    """An in-memory ordered component discovered from source blocks."""

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


class SourceStatement(BaseModel):
    """Persisted reconstructed source statement and ordered block membership."""

    uuid: str = Field(min_length=1)
    member_block_uuids: list[str] = Field(min_length=1)
    is_exercise: bool = False
    content: str = Field(min_length=1)
    embedding: list[float] = Field(min_length=1)


class SourceProcedure(BaseModel):
    """Persisted reconstructed source procedure and ordered block membership."""

    uuid: str = Field(min_length=1)
    member_block_uuids: list[str] = Field(min_length=1)
    content: str = Field(min_length=1)
    embedding: list[float] = Field(min_length=1)
