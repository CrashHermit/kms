"""Private shared fields for typed source-learning contracts."""

from pydantic import BaseModel, Field

from kms2.core.model.base import Vertex


class _LearningFactInput(BaseModel):
    """Model-facing fields for one learning-fact request."""

    hub_name: str = Field(min_length=1)
    source_fact_text: str = Field(min_length=1)


class _LearningFactVertex(Vertex):
    """Shared durable fields for typed source learning facts."""

    text: str = Field(min_length=1)


class _LearningFactRequest(BaseModel):
    """Backend identity and source content for one learning request."""

    hub_uuid: str = Field(min_length=1)
    hub_name: str = Field(min_length=1)
    source_fact_uuid: str = Field(min_length=1)
    source_fact_text: str = Field(min_length=1)


class _LearningFactCandidate(BaseModel):
    """Content-only learning-fact candidate returned by inference."""

    text: str = Field(min_length=1)


class _LearningFactOccurrence(BaseModel):
    """Durable learning fact with its originating graph identities."""

    hub_uuid: str = Field(min_length=1)
    source_fact_uuid: str = Field(min_length=1)


class _FlashcardInput(BaseModel):
    """Model-facing fields for one learning-fact card request."""

    learning_fact_text: str = Field(min_length=1)


class _FlashcardResult(BaseModel):
    """Content-only question and answer returned by card inference."""

    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
