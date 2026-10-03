"""Durable source flashcard model and persistence occurrence."""

from pydantic import BaseModel, Field

from kms2.core.model.base import Vertex


class SourceFlashcard(Vertex):
    """One source-owned flashcard derived from one learning fact."""

    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)


class SourceFlashcardOccurrence(BaseModel):
    """Persistable flashcard paired with its originating learning fact."""

    card: SourceFlashcard
    learning_fact_uuid: str = Field(min_length=1)
