"""Shared typed semantic description input model."""

from pydantic import BaseModel, Field

from kms2.core.model.context import SourceBlockContext


class TermDescriptionInput(BaseModel):
    """One typed occurrence and its source-local model context."""

    term: str = Field(min_length=1)
    context_before: list[SourceBlockContext] = Field(default_factory=list)
    target_block: SourceBlockContext
    context_after: list[SourceBlockContext] = Field(default_factory=list)


__all__ = ['TermDescriptionInput']
