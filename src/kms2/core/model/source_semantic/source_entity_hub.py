"""Source-local entity hub models."""

from typing import Annotated

from pydantic import BaseModel, Field

from kms2.core.model.base import Vertex

_NonEmptyString = Annotated[str, Field(min_length=1)]


class SourceEntityHubJudgeInput(BaseModel):
    """One ordered entity pair supplied to the membership judge."""

    index: int = Field(default=0, ge=0)
    left_name: str = Field(min_length=1)
    left_description: str = Field(min_length=1)
    right_name: str = Field(min_length=1)
    right_description: str = Field(min_length=1)


class SourceEntityHubJudgeDecision(BaseModel):
    """One indexed entity membership judgment."""

    index: int = Field(ge=0)
    belongs_in_same_hub: bool


class SourceEntityHubCandidate(BaseModel):
    """One canonical entity pair returned by vector candidate discovery."""

    left_uuid: str = Field(min_length=1)
    left_name: str = Field(min_length=1)
    left_description: str = Field(min_length=1)
    right_uuid: str = Field(min_length=1)
    right_name: str = Field(min_length=1)
    right_description: str = Field(min_length=1)
    score: float


class SourceEntityHubMember(BaseModel):
    """One entity occurrence returned by community detection."""

    uuid: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)


class SourceEntityHubSynthesisMember(BaseModel):
    """One entity occurrence supplied as synthesis evidence."""

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)


class SourceEntityHubSynthesisInput(BaseModel):
    """Entity community evidence supplied to the synthesis model."""

    members: list[SourceEntityHubSynthesisMember] = Field(min_length=1)


class SourceEntityHubDefinition(BaseModel):
    """Canonical definition synthesized for one entity community."""

    canonical_name: str = Field(min_length=1)
    description: str = Field(min_length=1)


class SourceEntityHub(Vertex):
    """Durable source-local entity hub vertex."""

    source_uuid: str = Field(min_length=1)
    canonical_name: str = Field(min_length=1)
    aliases: list[_NonEmptyString] = Field(default_factory=list)
    description: str = Field(min_length=1)
    embedding: list[float] = Field(min_length=1)


class SourceEntityHubRerankResult(BaseModel):
    """One ordered entity reranker batch result."""

    ordinal: int = Field(ge=0)
    direct: list[SourceEntityHubCandidate] = Field(default_factory=list)
    borderline: list[SourceEntityHubCandidate] = Field(default_factory=list)


class SourceEntityHubJudgeResult(BaseModel):
    """One ordered entity judge batch result."""

    ordinal: int = Field(ge=0)
    accepted: list[SourceEntityHubCandidate] = Field(default_factory=list)


class SourceEntityHubSynthesisResult(BaseModel):
    """One ordered entity community synthesis result."""

    ordinal: int = Field(ge=0)
    definition: SourceEntityHubDefinition
    membership_uuids: list[str] = Field(min_length=1)
    aliases: list[str] = Field(min_length=1)


__all__ = [
    'SourceEntityHub',
    'SourceEntityHubCandidate',
    'SourceEntityHubDefinition',
    'SourceEntityHubJudgeDecision',
    'SourceEntityHubJudgeInput',
    'SourceEntityHubJudgeResult',
    'SourceEntityHubMember',
    'SourceEntityHubRerankResult',
    'SourceEntityHubSynthesisInput',
    'SourceEntityHubSynthesisMember',
    'SourceEntityHubSynthesisResult',
]
