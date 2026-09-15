"""Source event occurrence and description models."""

from pydantic import BaseModel, Field

from kms2.core.model.semantic.base import _SemanticOccurrence
from kms2.core.model.semantic.term_description import TermDescriptionInput


class SourceEvent(_SemanticOccurrence):
    """Raw event occurrence attached to one source assertion."""

    name: str = Field(min_length=1)


class SourceEventDescriptionTarget(SourceEvent):
    """Raw event occurrence selected for description generation."""


class SourceEventDescriptionRequest(BaseModel):
    """One event occurrence and its description input."""

    target: SourceEventDescriptionTarget
    model_input: TermDescriptionInput


class SourceEventDescriptionResult(SourceEventDescriptionTarget):
    """One described event occurrence before graph persistence."""

    description: str = Field(min_length=1)
    embedding: list[float] | None = None


__all__ = [
    'SourceEvent',
    'SourceEventDescriptionRequest',
    'SourceEventDescriptionResult',
    'SourceEventDescriptionTarget',
]
