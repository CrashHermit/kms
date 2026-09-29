"""Global entity hub models."""

from typing import Annotated

from pydantic import BaseModel, Field

from kms2.core.model.base import Vertex

_NonEmptyString = Annotated[str, Field(min_length=1)]


class GlobalEntityHubJudgeInput(BaseModel):
    """One ordered entity pair supplied to the global membership judge."""

    index: int = Field(default=0, ge=0)
    left_canonical_name: str = Field(min_length=1)
    left_description: str = Field(min_length=1)
    right_canonical_name: str = Field(min_length=1)
    right_description: str = Field(min_length=1)


class GlobalEntityHubJudgeDecision(BaseModel):
    """One indexed global entity membership judgment."""

    index: int = Field(ge=0)
    belongs_in_same_hub: bool


class GlobalEntityHubCandidate(BaseModel):
    """One canonical source-hub pair returned by vector discovery."""

    left_uuid: str = Field(min_length=1)
    left_canonical_name: str = Field(min_length=1)
    left_description: str = Field(min_length=1)
    right_uuid: str = Field(min_length=1)
    right_canonical_name: str = Field(min_length=1)
    right_description: str = Field(min_length=1)
    score: float


class GlobalEntityHubMember(BaseModel):
    """One source entity hub returned by community detection."""

    uuid: str = Field(min_length=1)
    canonical_name: str = Field(min_length=1)
    description: str = Field(min_length=1)


class GlobalEntityHubSynthesisMember(BaseModel):
    """One source entity hub supplied as synthesis evidence."""

    canonical_name: str = Field(min_length=1)
    description: str = Field(min_length=1)


class GlobalEntityHubSynthesisInput(BaseModel):
    """Entity community evidence supplied to global synthesis."""

    members: list[GlobalEntityHubSynthesisMember] = Field(min_length=1)


class GlobalEntityHubDefinition(BaseModel):
    """Canonical definition synthesized for one global entity community."""

    canonical_name: str = Field(min_length=1)
    description: str = Field(min_length=1)


class GlobalEntityHub(Vertex):
    """Durable cross-source entity hub vertex."""

    canonical_name: str = Field(min_length=1)
    aliases: list[_NonEmptyString] = Field(default_factory=list)
    description: str = Field(min_length=1)
    embedding: list[float] = Field(min_length=1)


class GlobalEntityHubRerankResult(BaseModel):
    """One ordered global entity reranker batch result."""

    ordinal: int = Field(ge=0)
    direct: list[GlobalEntityHubCandidate] = Field(default_factory=list)
    borderline: list[GlobalEntityHubCandidate] = Field(default_factory=list)


class GlobalEntityHubJudgeResult(BaseModel):
    """One ordered global entity judge batch result."""

    ordinal: int = Field(ge=0)
    accepted: list[GlobalEntityHubCandidate] = Field(default_factory=list)


class GlobalEntityHubSynthesisResult(BaseModel):
    """One ordered global entity community synthesis result."""

    ordinal: int = Field(ge=0)
    definition: GlobalEntityHubDefinition
    membership_uuids: list[str] = Field(min_length=1)
    aliases: list[str] = Field(min_length=1)
