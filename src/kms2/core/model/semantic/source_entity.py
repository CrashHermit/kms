"""Source entity occurrence and description models."""

from pydantic import BaseModel, Field

from kms2.core.model.context import SourceBlockContext
from kms2.core.model.semantic.base import _SemanticOccurrence


class SourceEntity(_SemanticOccurrence):
    """Raw entity occurrence attached to one source assertion."""

    name: str = Field(min_length=1)


class SourceEntityDescriptionInput(BaseModel):
    """Entity occurrence and its source-local model context."""

    term: str = Field(min_length=1)
    context_before: list[SourceBlockContext] = Field(default_factory=list)
    target_block: SourceBlockContext
    context_after: list[SourceBlockContext] = Field(default_factory=list)


class SourceEntityDescriptionTarget(SourceEntity):
    """Raw entity occurrence selected for description generation."""


class SourceEntityDescriptionRequest(BaseModel):
    """One entity occurrence and its description input."""

    target: SourceEntityDescriptionTarget
    model_input: SourceEntityDescriptionInput


class SourceEntityDescriptionResult(SourceEntityDescriptionTarget):
    """One described entity occurrence before graph persistence."""

    description: str = Field(min_length=1)
    embedding: list[float] | None = None


__all__ = [
    'SourceEntity',
    'SourceEntityDescriptionInput',
    'SourceEntityDescriptionRequest',
    'SourceEntityDescriptionResult',
    'SourceEntityDescriptionTarget',
]
