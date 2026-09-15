"""Source predicate occurrence and description models."""

from pydantic import BaseModel, Field

from kms2.core.model.semantic.base import _SemanticOccurrence
from kms2.core.model.semantic.term_description import TermDescriptionInput


class SourcePredicate(_SemanticOccurrence):
    """Raw predicate occurrence attached to one source assertion."""

    predicate: str = Field(min_length=1)


class SourcePredicateDescriptionTarget(SourcePredicate):
    """Raw predicate occurrence selected for description generation."""


class SourcePredicateDescriptionRequest(BaseModel):
    """One predicate occurrence and its description input."""

    target: SourcePredicateDescriptionTarget
    model_input: TermDescriptionInput


class SourcePredicateDescriptionResult(SourcePredicateDescriptionTarget):
    """One described predicate occurrence before graph persistence."""

    description: str = Field(min_length=1)
    embedding: list[float] | None = None


__all__ = [
    'SourcePredicate',
    'SourcePredicateDescriptionRequest',
    'SourcePredicateDescriptionResult',
    'SourcePredicateDescriptionTarget',
]
