"""Global procedure hub models."""

from typing import Annotated

from pydantic import BaseModel, Field

from kms2.core.model.base import Vertex

_NonEmptyString = Annotated[str, Field(min_length=1)]


class GlobalProcedureHubJudgeInput(BaseModel):
    """One ordered procedure pair supplied to the global membership judge."""

    index: int = Field(default=0, ge=0)
    left_description: str = Field(min_length=1)
    right_description: str = Field(min_length=1)


class GlobalProcedureHubJudgeDecision(BaseModel):
    """One indexed global procedure membership judgment."""

    index: int = Field(ge=0)
    belongs_in_same_hub: bool


class GlobalProcedureHubCandidate(BaseModel):
    """One canonical source-hub pair returned by vector discovery."""

    left_uuid: str = Field(min_length=1)
    left_description: str = Field(min_length=1)
    right_uuid: str = Field(min_length=1)
    right_description: str = Field(min_length=1)
    score: float


class GlobalProcedureHubMember(BaseModel):
    """One source procedure hub returned by community detection."""

    uuid: str = Field(min_length=1)
    canonical_name: str = Field(min_length=1)
    description: str = Field(min_length=1)


class GlobalProcedureHubSynthesisMember(BaseModel):
    """One source procedure hub supplied as synthesis evidence."""

    description: str = Field(min_length=1)


class GlobalProcedureHubSynthesisInput(BaseModel):
    """Procedure community evidence supplied to global synthesis."""

    members: list[GlobalProcedureHubSynthesisMember] = Field(min_length=1)


class GlobalProcedureHubDefinition(BaseModel):
    """Canonical definition synthesized for one global procedure community."""

    canonical_name: str = Field(min_length=1)
    description: str = Field(min_length=1)


class GlobalProcedureHub(Vertex):
    """Durable cross-source procedure hub vertex."""

    canonical_name: str = Field(min_length=1)
    aliases: list[_NonEmptyString] = Field(default_factory=list)
    description: str = Field(min_length=1)
    embedding: list[float] = Field(min_length=1)


class GlobalProcedureHubRerankResult(BaseModel):
    """One ordered global procedure reranker batch result."""

    ordinal: int = Field(ge=0)
    direct: list[GlobalProcedureHubCandidate] = Field(default_factory=list)
    borderline: list[GlobalProcedureHubCandidate] = Field(default_factory=list)


class GlobalProcedureHubJudgeResult(BaseModel):
    """One ordered global procedure judge batch result."""

    ordinal: int = Field(ge=0)
    accepted: list[GlobalProcedureHubCandidate] = Field(default_factory=list)


class GlobalProcedureHubSynthesisResult(BaseModel):
    """One ordered global procedure community synthesis result."""

    ordinal: int = Field(ge=0)
    definition: GlobalProcedureHubDefinition
    membership_uuids: list[str] = Field(min_length=1)
    aliases: list[str] = Field(min_length=1)
