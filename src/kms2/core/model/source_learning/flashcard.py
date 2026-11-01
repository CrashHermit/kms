"""Durable source flashcards and their direct provenance."""

from pydantic import BaseModel, Field

from kms2.core.model.base import Vertex


class SourceFlashcard(Vertex):
    """One generic source-owned question and answer card."""

    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)


class SourceFlashcardCandidate(BaseModel):
    """Content-only question and answer returned by flashcard inference."""

    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)


class SourceFlashcardOccurrence(BaseModel):
    """Persistable card with selected source-block provenance."""

    card: SourceFlashcard
    source_block_uuids: list[str] = Field(min_length=1)
    derived_card_uuids: list[str] = Field(default_factory=list)
