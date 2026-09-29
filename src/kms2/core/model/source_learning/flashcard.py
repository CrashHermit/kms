"""Durable source flashcard model."""

from pydantic import Field

from kms2.core.model.base import Vertex


class SourceFlashcard(Vertex):
    """One source-owned flashcard derived from one learning fact."""

    source_uuid: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
