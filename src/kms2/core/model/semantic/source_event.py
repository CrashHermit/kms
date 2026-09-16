"""Source event occurrence and description models."""

from pydantic import BaseModel, Field

from kms2.core.model.context import SourceBlockContext
from kms2.core.model.semantic.base import _SemanticOccurrence


class SourceEvent(_SemanticOccurrence):
    """Raw event occurrence attached to one source assertion."""

    name: str = Field(min_length=1)


class SourceEventDescriptionInput(BaseModel):
    """Event occurrence and its source-local model context."""

    term: str = Field(min_length=1)
    context_before: list[SourceBlockContext] = Field(default_factory=list)
    target_block: SourceBlockContext
    context_after: list[SourceBlockContext] = Field(default_factory=list)


class SourceEventDescriptionTarget(SourceEvent):
    """Raw event occurrence selected for description generation."""


class SourceEventDescriptionRequest(BaseModel):
    """One event occurrence and its description input."""

    target: SourceEventDescriptionTarget
    model_input: SourceEventDescriptionInput


class SourceEventDescriptionResult(SourceEventDescriptionTarget):
    """One described event occurrence before graph persistence."""

    description: str = Field(min_length=1)
    embedding: list[float] | None = None


__all__ = [
    'SourceEvent',
    'SourceEventDescriptionInput',
    'SourceEventDescriptionRequest',
    'SourceEventDescriptionResult',
    'SourceEventDescriptionTarget',
]
