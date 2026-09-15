"""Ordered source page model."""

from pydantic import BaseModel, Field

from kms2.core.model.block import SourceBlock


class SourcePage(BaseModel):
    """An ordered source page containing markdown and canonical blocks."""

    index: int
    markdown: str = ''
    blocks: list[SourceBlock] = Field(default_factory=list)


__all__ = ['SourcePage']
