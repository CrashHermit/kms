"""Source-local entity hub models."""

from pydantic import BaseModel, Field

from kms.core.model.base import Vertex
from kms.core.model.context import SourceBlockWindow
from kms.core.model.source_extraction import SourceEvidenceInput


class SourceEntityHubJudgeInput(BaseModel):
    """One ordered identity judgment with UUID-free occurrence evidence."""

    index: int = Field(default=0, ge=0)
    left_name: str = Field(min_length=1)
    left_description: str = Field(min_length=1)
    left_evidence: SourceEvidenceInput
    right_name: str = Field(min_length=1)
    right_description: str = Field(min_length=1)
    right_evidence: SourceEvidenceInput


class SourceEntityHubJudgeDecision(BaseModel):
    """One indexed entity identity decision."""

    index: int = Field(ge=0)
    belongs_in_same_hub: bool


class SourceEntityHubCandidate(BaseModel):
    """One occurrence pair proposed across two current groups."""

    left_group_uuid: str = Field(min_length=1)
    right_group_uuid: str = Field(min_length=1)
    left_uuid: str = Field(min_length=1)
    left_name: str = Field(min_length=1)
    left_description: str = Field(min_length=1)
    left_window: SourceBlockWindow
    right_uuid: str = Field(min_length=1)
    right_name: str = Field(min_length=1)
    right_description: str = Field(min_length=1)
    right_window: SourceBlockWindow
    score: float


class SourceEntityHubMember(BaseModel):
    """An entity occurrence initialized with its canonical owning window."""

    uuid: str = Field(min_length=1)
    source_block_uuid: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    window: SourceBlockWindow | None = None


class SourceEntityHub(Vertex):
    """Durable source-local entity hub identity."""


class SourceEntityHubRerankResult(BaseModel):
    """One ordered entity reranker batch result."""

    ordinal: int = Field(ge=0)
    survivors: list[SourceEntityHubCandidate] = Field(default_factory=list)
    rejected: list[SourceEntityHubCandidate] = Field(default_factory=list)


class SourceEntityHubJudgeResult(BaseModel):
    """One ordered entity judge batch result."""

    ordinal: int = Field(ge=0)
    accepted: list[SourceEntityHubCandidate] = Field(default_factory=list)
    rejected: list[SourceEntityHubCandidate] = Field(default_factory=list)
