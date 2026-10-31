"""Source-local proposition hub models."""

from pydantic import BaseModel, Field, model_validator

from kms2.core.model.base import Vertex


class SourcePropositionHubMember(BaseModel):
    """One proposition occurrence available for source-local grouping."""

    uuid: str = Field(min_length=1)
    text: str = Field(min_length=1)


class SourcePropositionHubCandidate(BaseModel):
    """One canonical pair returned by proposition vector discovery."""

    left_uuid: str = Field(min_length=1)
    left_text: str = Field(min_length=1)
    right_uuid: str = Field(min_length=1)
    right_text: str = Field(min_length=1)
    score: float

    @model_validator(mode='after')
    def validate_uuid_order(self) -> SourcePropositionHubCandidate:
        """Keep each unordered pair in canonical UUID order."""
        if self.left_uuid >= self.right_uuid:
            raise ValueError('left_uuid must sort before right_uuid')
        return self


class SourcePropositionHubJudgeInput(BaseModel):
    """One proposition pair supplied to the equivalence judge."""

    index: int = Field(ge=0)
    left_text: str = Field(min_length=1)
    right_text: str = Field(min_length=1)


class SourcePropositionHubJudgeDecision(BaseModel):
    """One indexed proposition-equivalence judgment."""

    index: int = Field(ge=0)
    equivalent: bool


class SourcePropositionHub(Vertex):
    """Identity-only source-local hub for equivalent proposition occurrences."""


class SourcePropositionHubRerankResult(BaseModel):
    """One indexed reranker batch result."""

    ordinal: int = Field(ge=0)
    candidates: list[SourcePropositionHubCandidate] = Field(
        default_factory=list
    )


class SourcePropositionHubJudgeResult(BaseModel):
    """One indexed equivalence-judge batch result."""

    ordinal: int = Field(ge=0)
    accepted: list[SourcePropositionHubCandidate] = Field(default_factory=list)


class SourcePropositionHubGroup(BaseModel):
    """One disjoint proposition group in source order."""

    ordinal: int = Field(ge=0)
    members: list[SourcePropositionHubMember] = Field(min_length=1)
