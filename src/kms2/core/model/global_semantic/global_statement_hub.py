"""Global statement hub models."""

from typing import Annotated

from pydantic import BaseModel, Field

from kms2.core.model.base import Vertex

_NonEmptyString = Annotated[str, Field(min_length=1)]


class GlobalStatementHubJudgeInput(BaseModel):
    """One ordered statement pair supplied to the global membership judge."""

    index: int = Field(default=0, ge=0)
    left_description: str = Field(min_length=1)
    right_description: str = Field(min_length=1)


class GlobalStatementHubJudgeDecision(BaseModel):
    """One indexed global statement membership judgment."""

    index: int = Field(ge=0)
    belongs_in_same_hub: bool


class GlobalStatementHubCandidate(BaseModel):
    """One canonical source-hub pair returned by vector discovery."""

    left_uuid: str = Field(min_length=1)
    left_description: str = Field(min_length=1)
    right_uuid: str = Field(min_length=1)
    right_description: str = Field(min_length=1)
    score: float


class GlobalStatementHubMember(BaseModel):
    """One source statement hub returned by community detection."""

    uuid: str = Field(min_length=1)
    canonical_name: str = Field(min_length=1)
    description: str = Field(min_length=1)


class GlobalStatementHubSynthesisMember(BaseModel):
    """One source statement hub supplied as synthesis evidence."""

    description: str = Field(min_length=1)


class GlobalStatementHubSynthesisInput(BaseModel):
    """Statement community evidence supplied to global synthesis."""

    members: list[GlobalStatementHubSynthesisMember] = Field(min_length=1)


class GlobalStatementHubSummary(BaseModel):
    """Temporary statement evidence summary, not a hub vertex."""

    text: str = Field(min_length=1)


class GlobalStatementHubSummaryInput(BaseModel):
    """Statement evidence supplied for one summary."""

    evidence: list[str] = Field(min_length=1)


class GlobalStatementHubSummaryMergeInput(BaseModel):
    """Ordered statement summaries supplied for one merge."""

    summaries: list[GlobalStatementHubSummary] = Field(min_length=2)


class GlobalStatementHubSummarySynthesisInput(BaseModel):
    """Ordered statement summaries supplied to final synthesis."""

    summaries: list[GlobalStatementHubSummary] = Field(min_length=1)


class GlobalStatementHubDefinition(BaseModel):
    """Canonical definition synthesized for one global statement community."""

    canonical_name: str = Field(min_length=1)
    description: str = Field(min_length=1)


class GlobalStatementHub(Vertex):
    """Durable cross-source statement hub vertex."""

    canonical_name: str = Field(min_length=1)
    aliases: list[_NonEmptyString] = Field(default_factory=list)
    description: str = Field(min_length=1)
    embedding: list[float] = Field(min_length=1)


class GlobalStatementHubRerankResult(BaseModel):
    """One ordered global statement reranker batch result."""

    ordinal: int = Field(ge=0)
    direct: list[GlobalStatementHubCandidate] = Field(default_factory=list)
    borderline: list[GlobalStatementHubCandidate] = Field(default_factory=list)


class GlobalStatementHubJudgeResult(BaseModel):
    """One ordered global statement judge batch result."""

    ordinal: int = Field(ge=0)
    accepted: list[GlobalStatementHubCandidate] = Field(default_factory=list)


class GlobalStatementHubSynthesisResult(BaseModel):
    """One ordered global statement community synthesis result."""

    ordinal: int = Field(ge=0)
    definition: GlobalStatementHubDefinition
    membership_uuids: list[str] = Field(min_length=1)
    aliases: list[str] = Field(min_length=1)
