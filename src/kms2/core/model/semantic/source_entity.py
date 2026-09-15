"""Source entity occurrence and description models."""

from pydantic import BaseModel, Field

from kms2.core.model.semantic.base import _SemanticOccurrence
from kms2.core.model.semantic.term_description import TermDescriptionInput


class SourceEntity(_SemanticOccurrence):
    """Raw entity occurrence attached to one source assertion."""

    name: str = Field(min_length=1)


class SourceEntityDescriptionTarget(SourceEntity):
    """Raw entity occurrence selected for description generation."""


class SourceEntityDescriptionRequest(BaseModel):
    """One entity occurrence and its description input."""

    target: SourceEntityDescriptionTarget
    model_input: TermDescriptionInput


class SourceEntityDescriptionResult(SourceEntityDescriptionTarget):
    """One described entity occurrence before graph persistence."""

    description: str = Field(min_length=1)
    embedding: list[float] | None = None


__all__ = [
    'SourceEntity',
    'SourceEntityDescriptionRequest',
    'SourceEntityDescriptionResult',
    'SourceEntityDescriptionTarget',
]
