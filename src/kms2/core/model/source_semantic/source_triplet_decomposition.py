"""Triplet decomposition request, result, and candidate models."""

from enum import StrEnum

from pydantic import BaseModel, Field

from kms2.core.model.source_semantic.source_fact_extraction import SourceFact


class SourceTripletEndpointKind(StrEnum):
    """Endpoint kinds emitted by source-level triplet decomposition."""

    ENTITY = 'entity'
    EVENT = 'event'


class SourceTripletDecompositionCandidate(BaseModel):
    """One source-level subject-predicate-object decomposition candidate."""

    subject: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object: str = Field(min_length=1)
    subject_kind: SourceTripletEndpointKind
    object_kind: SourceTripletEndpointKind


class SourceTripletDecompositionRequest(BaseModel):
    """One persisted source fact supplied to the second model pass."""

    fact: SourceFact

    @property
    def fact_text(self) -> str:
        """Return the source-faithful fact text for model input."""
        return self.fact.text


class SourceTripletDecompositionResult(BaseModel):
    """Triplet candidates retaining the originating durable source fact."""

    fact: SourceFact
    triplets: list[SourceTripletDecompositionCandidate] = Field(
        default_factory=list
    )
