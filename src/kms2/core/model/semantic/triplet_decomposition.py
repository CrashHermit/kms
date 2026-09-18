"""Triplet decomposition request, result, and candidate models."""

from enum import StrEnum

from pydantic import BaseModel, Field

from kms2.core.model.semantic.fact_extraction import ExtractedFact


class TripletEndpointKind(StrEnum):
    """Endpoint kinds emitted by source-level triplet decomposition."""

    ENTITY = 'entity'
    EVENT = 'event'


class TripletDecompositionCandidate(BaseModel):
    """One source-level subject-predicate-object decomposition candidate."""

    subject: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object: str = Field(min_length=1)
    subject_kind: TripletEndpointKind
    object_kind: TripletEndpointKind


class TripletDecompositionRequest(BaseModel):
    """One extracted fact supplied to the second model pass."""

    fact_index: int = Field(default=0, ge=0)
    fact: ExtractedFact

    @property
    def fact_text(self) -> str:
        """Return the source-faithful fact text for model input."""
        return self.fact.text


class TripletDecompositionResult(BaseModel):
    """Triplet candidates retaining the originating fact provenance."""

    fact_index: int = Field(default=0, ge=0)
    fact: ExtractedFact
    triplets: list[TripletDecompositionCandidate] = Field(default_factory=list)


__all__ = [
    'TripletDecompositionCandidate',
    'TripletDecompositionRequest',
    'TripletDecompositionResult',
    'TripletEndpointKind',
]
