"""Source procedure description transport models."""

from pydantic import BaseModel, Field

from kms2.core.model.context import SourceBlockContext


class SourceProcedure(BaseModel):
    """Persisted source procedure identity and ordered block membership."""

    uuid: str = Field(min_length=1)
    source_uuid: str = Field(min_length=1)
    member_block_uuids: list[str] = Field(min_length=1)


class SourceProcedureDescriptionInput(BaseModel):
    """Procedure blocks and source-local context supplied to the model."""

    context_before: list[SourceBlockContext] = Field(default_factory=list)
    target_blocks: list[SourceBlockContext] = Field(min_length=1)
    context_after: list[SourceBlockContext] = Field(default_factory=list)


class SourceProcedureDescriptionTarget(SourceProcedure):
    """Source procedure selected for description generation."""


class SourceProcedureDescriptionRequest(BaseModel):
    """One source procedure and its description input."""

    target: SourceProcedureDescriptionTarget
    model_input: SourceProcedureDescriptionInput


class SourceProcedureDescriptionResult(SourceProcedureDescriptionTarget):
    """Described source procedure before embedding persistence."""

    description: str = Field(min_length=1)
    embedding: list[float] | None = None


__all__ = [
    'SourceProcedure',
    'SourceProcedureDescriptionInput',
    'SourceProcedureDescriptionRequest',
    'SourceProcedureDescriptionResult',
    'SourceProcedureDescriptionTarget',
]
