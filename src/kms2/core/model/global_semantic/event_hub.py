"""Global event hub models."""

from typing import Annotated

from pydantic import BaseModel, Field

from kms2.core.model.base import Vertex

_NonEmptyString = Annotated[str, Field(min_length=1)]


class GlobalEventHubJudgeInput(BaseModel):
    """One ordered event pair supplied to the global membership judge."""

    index: int = Field(default=0, ge=0)
    left_name: str = Field(min_length=1)
    left_description: str = Field(min_length=1)
    right_name: str = Field(min_length=1)
    right_description: str = Field(min_length=1)


class GlobalEventHubJudgeDecision(BaseModel):
    """One indexed global event membership judgment."""

    index: int = Field(ge=0)
    belongs_in_same_hub: bool


class GlobalEventHubCandidate(BaseModel):
    """One canonical source-hub pair returned by vector discovery."""

    left_uuid: str = Field(min_length=1)
    left_name: str = Field(min_length=1)
    left_description: str = Field(min_length=1)
    right_uuid: str = Field(min_length=1)
    right_name: str = Field(min_length=1)
    right_description: str = Field(min_length=1)
    score: float


class GlobalEventHubMember(BaseModel):
    """One source event hub returned by community detection."""

    uuid: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)


class GlobalEventHubSynthesisMember(BaseModel):
    """One source event hub supplied as synthesis evidence."""

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)


class GlobalEventHubSynthesisInput(BaseModel):
    """Event community evidence supplied to global synthesis."""

    members: list[GlobalEventHubSynthesisMember] = Field(min_length=1)


class GlobalEventHubDefinition(BaseModel):
    """Canonical definition synthesized for one global event community."""

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)


class GlobalEventHub(Vertex):
    """Durable cross-source event hub vertex."""

    name: str = Field(min_length=1)
    aliases: list[_NonEmptyString] = Field(default_factory=list)
    description: str = Field(min_length=1)
    embedding: list[float] = Field(min_length=1)


class GlobalEventHubRerankResult(BaseModel):
    """One ordered global event reranker batch result."""

    ordinal: int = Field(ge=0)
    direct: list[GlobalEventHubCandidate] = Field(default_factory=list)
    borderline: list[GlobalEventHubCandidate] = Field(default_factory=list)


class GlobalEventHubJudgeResult(BaseModel):
    """One ordered global event judge batch result."""

    ordinal: int = Field(ge=0)
    accepted: list[GlobalEventHubCandidate] = Field(default_factory=list)


class GlobalEventHubSynthesisResult(BaseModel):
    """One ordered global event community synthesis result."""

    ordinal: int = Field(ge=0)
    definition: GlobalEventHubDefinition
    membership_uuids: list[str] = Field(min_length=1)
    aliases: list[str] = Field(min_length=1)


__all__ = [
    'GlobalEventHub',
    'GlobalEventHubCandidate',
    'GlobalEventHubDefinition',
    'GlobalEventHubJudgeDecision',
    'GlobalEventHubJudgeInput',
    'GlobalEventHubJudgeResult',
    'GlobalEventHubMember',
    'GlobalEventHubRerankResult',
    'GlobalEventHubSynthesisInput',
    'GlobalEventHubSynthesisMember',
    'GlobalEventHubSynthesisResult',
]
