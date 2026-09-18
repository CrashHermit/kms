"""Source-local procedure hub models."""

from typing import Annotated

from pydantic import BaseModel, Field

from kms2.core.model.base import Vertex

_NonEmptyString = Annotated[str, Field(min_length=1)]


class SourceProcedureHubJudgeInput(BaseModel):
    """One ordered procedure pair supplied to the membership judge."""

    index: int = Field(default=0, ge=0)
    left_description: str = Field(min_length=1)
    right_description: str = Field(min_length=1)


class SourceProcedureHubJudgeDecision(BaseModel):
    """One indexed procedure membership judgment."""

    index: int = Field(ge=0)
    belongs_in_same_hub: bool


class SourceProcedureHubCandidate(BaseModel):
    """One canonical procedure pair returned by vector candidate discovery."""

    left_uuid: str = Field(min_length=1)
    left_description: str = Field(min_length=1)
    right_uuid: str = Field(min_length=1)
    right_description: str = Field(min_length=1)
    score: float


class SourceProcedureHubMember(BaseModel):
    """One procedure occurrence returned by community detection."""

    uuid: str = Field(min_length=1)
    description: str = Field(min_length=1)


class SourceProcedureHubSynthesisMember(BaseModel):
    """One procedure occurrence supplied as synthesis evidence."""

    description: str = Field(min_length=1)


class SourceProcedureHubSynthesisInput(BaseModel):
    """Procedure community evidence supplied to the synthesis model."""

    members: list[SourceProcedureHubSynthesisMember] = Field(min_length=1)


class SourceProcedureHubDefinition(BaseModel):
    """Canonical definition synthesized for one procedure community."""

    canonical_name: str = Field(min_length=1)
    description: str = Field(min_length=1)


class SourceProcedureHub(Vertex):
    """Durable source-local procedure hub vertex."""

    source_uuid: str = Field(min_length=1)
    canonical_name: str = Field(min_length=1)
    aliases: list[_NonEmptyString] = Field(default_factory=list)
    description: str = Field(min_length=1)
    embedding: list[float] = Field(min_length=1)


class SourceProcedureHubRerankResult(BaseModel):
    """One ordered procedure reranker batch result."""

    ordinal: int = Field(ge=0)
    direct: list[SourceProcedureHubCandidate] = Field(default_factory=list)
    borderline: list[SourceProcedureHubCandidate] = Field(default_factory=list)


class SourceProcedureHubJudgeResult(BaseModel):
    """One ordered procedure judge batch result."""

    ordinal: int = Field(ge=0)
    accepted: list[SourceProcedureHubCandidate] = Field(default_factory=list)


class SourceProcedureHubSynthesisResult(BaseModel):
    """One ordered procedure community synthesis result."""

    ordinal: int = Field(ge=0)
    definition: SourceProcedureHubDefinition
    membership_uuids: list[str] = Field(min_length=1)


__all__ = [
    'SourceProcedureHub',
    'SourceProcedureHubCandidate',
    'SourceProcedureHubDefinition',
    'SourceProcedureHubJudgeDecision',
    'SourceProcedureHubJudgeInput',
    'SourceProcedureHubJudgeResult',
    'SourceProcedureHubMember',
    'SourceProcedureHubRerankResult',
    'SourceProcedureHubSynthesisInput',
    'SourceProcedureHubSynthesisMember',
    'SourceProcedureHubSynthesisResult',
]
