"""Source-local statement hub models."""

from typing import Annotated

from pydantic import BaseModel, Field

from kms2.core.model.base import Vertex

_NonEmptyString = Annotated[str, Field(min_length=1)]


class SourceStatementHubJudgeInput(BaseModel):
    """One ordered statement pair supplied to the membership judge."""

    index: int = Field(default=0, ge=0)
    left_description: str = Field(min_length=1)
    right_description: str = Field(min_length=1)


class SourceStatementHubJudgeDecision(BaseModel):
    """One indexed statement membership judgment."""

    index: int = Field(ge=0)
    belongs_in_same_hub: bool


class SourceStatementHubCandidate(BaseModel):
    """One canonical statement pair returned by vector candidate discovery."""

    left_uuid: str = Field(min_length=1)
    left_description: str = Field(min_length=1)
    right_uuid: str = Field(min_length=1)
    right_description: str = Field(min_length=1)
    score: float


class SourceStatementHubMember(BaseModel):
    """One statement occurrence returned by community detection."""

    uuid: str = Field(min_length=1)
    description: str = Field(min_length=1)


class SourceStatementHubSynthesisMember(BaseModel):
    """One statement occurrence supplied as synthesis evidence."""

    description: str = Field(min_length=1)


class SourceStatementHubSynthesisInput(BaseModel):
    """Statement community evidence supplied to the synthesis model."""

    members: list[SourceStatementHubSynthesisMember] = Field(min_length=1)


class SourceStatementHubDefinition(BaseModel):
    """Canonical definition synthesized for one statement community."""

    canonical_name: str = Field(min_length=1)
    description: str = Field(min_length=1)


class SourceStatementHub(Vertex):
    """Durable source-local statement hub vertex."""

    source_uuid: str = Field(min_length=1)
    canonical_name: str = Field(min_length=1)
    aliases: list[_NonEmptyString] = Field(default_factory=list)
    description: str = Field(min_length=1)
    embedding: list[float] = Field(min_length=1)


__all__ = [
    'SourceStatementHub',
    'SourceStatementHubCandidate',
    'SourceStatementHubDefinition',
    'SourceStatementHubJudgeDecision',
    'SourceStatementHubJudgeInput',
    'SourceStatementHubMember',
    'SourceStatementHubSynthesisInput',
    'SourceStatementHubSynthesisMember',
]
