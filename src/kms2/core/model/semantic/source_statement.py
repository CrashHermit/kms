"""Source statement description transport models."""

from pydantic import BaseModel, Field

from kms2.core.model.context import SourceBlockContext


class SourceStatement(BaseModel):
    """Persisted source statement identity and ordered block membership."""

    uuid: str = Field(min_length=1)
    source_uuid: str = Field(min_length=1)
    member_block_uuids: list[str] = Field(min_length=1)
    is_exercise: bool = False


class SourceStatementDescriptionInput(BaseModel):
    """Statement blocks and source-local context supplied to the model."""

    context_before: list[SourceBlockContext] = Field(default_factory=list)
    target_blocks: list[SourceBlockContext] = Field(min_length=1)
    context_after: list[SourceBlockContext] = Field(default_factory=list)


class SourceStatementDescriptionTarget(SourceStatement):
    """Source statement selected for description generation."""


class SourceStatementDescriptionRequest(BaseModel):
    """One source statement and its description input."""

    target: SourceStatementDescriptionTarget
    model_input: SourceStatementDescriptionInput


class SourceStatementDescriptionResult(SourceStatementDescriptionTarget):
    """Described source statement before embedding persistence."""

    description: str = Field(min_length=1)
    embedding: list[float] | None = None


__all__ = [
    'SourceStatement',
    'SourceStatementDescriptionInput',
    'SourceStatementDescriptionRequest',
    'SourceStatementDescriptionResult',
    'SourceStatementDescriptionTarget',
]
