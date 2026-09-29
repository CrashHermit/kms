"""Global predicate hub models."""

from typing import Annotated

from pydantic import BaseModel, Field

from kms2.core.model.base import Vertex

_NonEmptyString = Annotated[str, Field(min_length=1)]


class GlobalPredicateHubJudgeInput(BaseModel):
    """One ordered predicate pair supplied to the global membership judge."""

    index: int = Field(default=0, ge=0)
    left_predicate: str = Field(min_length=1)
    left_description: str = Field(min_length=1)
    right_predicate: str = Field(min_length=1)
    right_description: str = Field(min_length=1)


class GlobalPredicateHubJudgeDecision(BaseModel):
    """One indexed global predicate membership judgment."""

    index: int = Field(ge=0)
    belongs_in_same_hub: bool


class GlobalPredicateHubCandidate(BaseModel):
    """One canonical source-hub pair returned by vector discovery."""

    left_uuid: str = Field(min_length=1)
    left_predicate: str = Field(min_length=1)
    left_description: str = Field(min_length=1)
    right_uuid: str = Field(min_length=1)
    right_predicate: str = Field(min_length=1)
    right_description: str = Field(min_length=1)
    score: float


class GlobalPredicateHubMember(BaseModel):
    """One source predicate hub returned by community detection."""

    uuid: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    description: str = Field(min_length=1)


class GlobalPredicateHubSynthesisMember(BaseModel):
    """One source predicate hub supplied as synthesis evidence."""

    predicate: str = Field(min_length=1)
    description: str = Field(min_length=1)


class GlobalPredicateHubSynthesisInput(BaseModel):
    """Predicate community evidence supplied to global synthesis."""

    members: list[GlobalPredicateHubSynthesisMember] = Field(min_length=1)


class GlobalPredicateHubDefinition(BaseModel):
    """Canonical definition synthesized for one global predicate community."""

    predicate: str = Field(min_length=1)
    description: str = Field(min_length=1)


class GlobalPredicateHub(Vertex):
    """Durable cross-source predicate hub vertex."""

    predicate: str = Field(min_length=1)
    aliases: list[_NonEmptyString] = Field(default_factory=list)
    description: str = Field(min_length=1)
    embedding: list[float] = Field(min_length=1)


class GlobalPredicateHubRerankResult(BaseModel):
    """One ordered global predicate reranker batch result."""

    ordinal: int = Field(ge=0)
    direct: list[GlobalPredicateHubCandidate] = Field(default_factory=list)
    borderline: list[GlobalPredicateHubCandidate] = Field(default_factory=list)


class GlobalPredicateHubJudgeResult(BaseModel):
    """One ordered global predicate judge batch result."""

    ordinal: int = Field(ge=0)
    accepted: list[GlobalPredicateHubCandidate] = Field(default_factory=list)


class GlobalPredicateHubSynthesisResult(BaseModel):
    """One ordered global predicate community synthesis result."""

    ordinal: int = Field(ge=0)
    definition: GlobalPredicateHubDefinition
    membership_uuids: list[str] = Field(min_length=1)
    aliases: list[str] = Field(min_length=1)
