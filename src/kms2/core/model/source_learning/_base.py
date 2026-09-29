"""Private shared fields for typed source-learning contracts."""

from pydantic import BaseModel, Field

from kms2.core.model.base import Vertex


class _LearningFactEvidence(BaseModel):
    """One source fact and its complete directed triplet context."""

    source_fact_uuid: str = Field(min_length=1)
    source_fact_text: str = Field(min_length=1)
    triplet_uuid: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object: str = Field(min_length=1)
    member_role: str = Field(min_length=1)


class _LearningFactInput(BaseModel):
    """Shared shape for one typed hub's learning-fact evidence."""

    hub_uuid: str = Field(min_length=1)
    hub_name: str = Field(min_length=1)
    hub_description: str = Field(min_length=1)
    evidence: list[_LearningFactEvidence] = Field(min_length=1)


class _LearningFactOutput(BaseModel):
    """Shared atomic learning-fact output shape."""

    text: str = Field(min_length=1)
    source_fact_uuids: list[str] = Field(min_length=1)
    triplet_uuids: list[str] = Field(min_length=1)


class _LearningFactResult(BaseModel):
    """Shared collection returned by one learning-fact inference."""

    facts: list[_LearningFactOutput] = Field(default_factory=list)


class _FlashcardDraft(BaseModel):
    """One synthesized question and answer."""

    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)


class _FlashcardResult(_FlashcardDraft):
    """Validated one-card inference result."""


class _FlashcardInput(BaseModel):
    """Shared shape for one learning fact's card prompt."""

    learning_fact_text: str = Field(min_length=1)
    hub_uuid: str = Field(min_length=1)
    hub_name: str = Field(min_length=1)
    hub_description: str = Field(min_length=1)
    evidence: list[_LearningFactEvidence] = Field(min_length=1)


class _LearningFactVertex(Vertex):
    """Shared durable fields for typed source learning facts."""

    source_uuid: str = Field(min_length=1)
    text: str = Field(min_length=1)
