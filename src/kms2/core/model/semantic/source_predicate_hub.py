"""Source-local predicate hub models."""

from typing import Annotated

from pydantic import BaseModel, Field

from kms2.core.model.base import Vertex

_NonEmptyString = Annotated[str, Field(min_length=1)]


class SourcePredicateHubJudgeInput(BaseModel):
    """One ordered directed-relation pair supplied to the membership judge."""

    index: int = Field(default=0, ge=0)
    left_subject: str = Field(min_length=1)
    left_predicate: str = Field(min_length=1)
    left_object: str = Field(min_length=1)
    left_description: str = Field(min_length=1)
    right_subject: str = Field(min_length=1)
    right_predicate: str = Field(min_length=1)
    right_object: str = Field(min_length=1)
    right_description: str = Field(min_length=1)


class SourcePredicateHubJudgeDecision(BaseModel):
    """One indexed predicate membership judgment."""

    index: int = Field(ge=0)
    belongs_in_same_hub: bool


class SourcePredicateHubCandidate(BaseModel):
    """One canonical predicate pair returned by vector candidate discovery."""

    left_uuid: str = Field(min_length=1)
    left_predicate: str = Field(min_length=1)
    left_description: str = Field(min_length=1)
    left_subject: str = Field(min_length=1)
    left_object: str = Field(min_length=1)
    right_uuid: str = Field(min_length=1)
    right_predicate: str = Field(min_length=1)
    right_description: str = Field(min_length=1)
    right_subject: str = Field(min_length=1)
    right_object: str = Field(min_length=1)
    score: float


class SourcePredicateHubMember(BaseModel):
    """One predicate occurrence returned by community detection."""

    uuid: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    description: str = Field(min_length=1)


class SourcePredicateHubSynthesisMember(BaseModel):
    """One predicate occurrence supplied as synthesis evidence."""

    predicate: str = Field(min_length=1)
    description: str = Field(min_length=1)


class SourcePredicateHubSynthesisInput(BaseModel):
    """Predicate community evidence supplied to the synthesis model."""

    members: list[SourcePredicateHubSynthesisMember] = Field(min_length=1)


class SourcePredicateHubDefinition(BaseModel):
    """Canonical definition synthesized for one predicate community."""

    predicate: str = Field(min_length=1)
    description: str = Field(min_length=1)


class SourcePredicateHub(Vertex):
    """Durable source-local predicate hub vertex."""

    source_uuid: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    aliases: list[_NonEmptyString] = Field(default_factory=list)
    description: str = Field(min_length=1)
    embedding: list[float] = Field(min_length=1)


__all__ = [
    'SourcePredicateHub',
    'SourcePredicateHubCandidate',
    'SourcePredicateHubDefinition',
    'SourcePredicateHubJudgeInput',
    'SourcePredicateHubMember',
    'SourcePredicateHubSynthesisInput',
    'SourcePredicateHubSynthesisMember',
]
