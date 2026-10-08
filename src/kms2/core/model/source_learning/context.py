"""Content-only context shared by source-learning inference passes."""

from pydantic import BaseModel, Field

from kms2.core.model.source_semantic.source_triplet_hub import (
    SourceTripletHubRole,
)


class SourceLearningHubContext(BaseModel):
    """The persisted triplet hub and its three role-hub descriptions."""

    triplet_hub: SourceTripletHubRole
    subject_hub: SourceTripletHubRole
    predicate_hub: SourceTripletHubRole
    object_hub: SourceTripletHubRole


class SourceLearningEvidence(BaseModel):
    """One exact directed triplet and its originating source-fact text."""

    subject: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object: str = Field(min_length=1)
    source_fact_text: str = Field(min_length=1)
