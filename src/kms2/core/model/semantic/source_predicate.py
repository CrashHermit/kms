"""Source predicate occurrence and description models."""

from pydantic import BaseModel, Field

from kms2.core.model.context import SourceBlockContext
from kms2.core.model.semantic.base import _SemanticOccurrence


class SourcePredicate(_SemanticOccurrence):
    """Source-scoped predicate occurrence attached to one triplet."""

    predicate: str = Field(min_length=1)


class SourcePredicateDescriptionInput(BaseModel):
    """Predicate occurrence and its source-local model context."""

    term: str = Field(min_length=1)
    context_before: list[SourceBlockContext] = Field(default_factory=list)
    target_block: SourceBlockContext
    context_after: list[SourceBlockContext] = Field(default_factory=list)


class SourcePredicateDescriptionTarget(SourcePredicate):
    """Raw predicate occurrence selected for description generation."""


class SourcePredicateDescriptionRequest(BaseModel):
    """One predicate occurrence and its description input."""

    target: SourcePredicateDescriptionTarget
    model_input: SourcePredicateDescriptionInput


class SourcePredicateDescriptionResult(SourcePredicateDescriptionTarget):
    """One described predicate occurrence before graph persistence."""

    description: str = Field(min_length=1)
    embedding: list[float] | None = None


__all__ = [
    'SourcePredicate',
    'SourcePredicateDescriptionInput',
    'SourcePredicateDescriptionRequest',
    'SourcePredicateDescriptionResult',
    'SourcePredicateDescriptionTarget',
]
