"""Source-local event hub models."""

from typing import Annotated

from pydantic import BaseModel, Field

from kms2.core.model.base import Vertex

_NonEmptyString = Annotated[str, Field(min_length=1)]


class SourceEventHubJudgeInput(BaseModel):
    """One ordered event pair supplied to the membership judge."""

    index: int = Field(default=0, ge=0)
    left_name: str = Field(min_length=1)
    left_description: str = Field(min_length=1)
    right_name: str = Field(min_length=1)
    right_description: str = Field(min_length=1)


class SourceEventHubJudgeDecision(BaseModel):
    """One indexed event membership judgment."""

    index: int = Field(ge=0)
    belongs_in_same_hub: bool


class SourceEventHubCandidate(BaseModel):
    """One canonical event pair returned by vector candidate discovery."""

    left_uuid: str = Field(min_length=1)
    left_name: str = Field(min_length=1)
    left_description: str = Field(min_length=1)
    right_uuid: str = Field(min_length=1)
    right_name: str = Field(min_length=1)
    right_description: str = Field(min_length=1)
    score: float


class SourceEventHubMember(BaseModel):
    """One event occurrence returned by community detection."""

    uuid: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)


class SourceEventHubSynthesisMember(BaseModel):
    """One event occurrence supplied as synthesis evidence."""

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)


class SourceEventHubSynthesisInput(BaseModel):
    """Event community evidence supplied to the synthesis model."""

    members: list[SourceEventHubSynthesisMember] = Field(min_length=1)


class SourceEventHubDefinition(BaseModel):
    """Canonical definition synthesized for one event community."""

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)


class SourceEventHub(Vertex):
    """Durable source-local event hub vertex."""

    source_uuid: str = Field(min_length=1)
    name: str = Field(min_length=1)
    aliases: list[_NonEmptyString] = Field(default_factory=list)
    description: str = Field(min_length=1)
    embedding: list[float] = Field(min_length=1)


__all__ = [
    'SourceEventHub',
    'SourceEventHubCandidate',
    'SourceEventHubDefinition',
    'SourceEventHubJudgeInput',
    'SourceEventHubMember',
    'SourceEventHubSynthesisInput',
    'SourceEventHubSynthesisMember',
]
