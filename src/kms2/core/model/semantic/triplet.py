"""Triplet decomposition and raw assertion models."""

from enum import StrEnum

from pydantic import BaseModel, Field

from kms2.core.model.semantic.base import _SemanticOccurrence
from kms2.core.model.semantic.fact_extraction import ExtractedFact
from kms2.core.model.semantic.source_entity import SourceEntity
from kms2.core.model.semantic.source_event import SourceEvent
from kms2.core.model.semantic.source_predicate import SourcePredicate


class SemanticNodeKind(StrEnum):
    """Endpoint kinds emitted by source-level triplet decomposition."""

    ENTITY = 'entity'
    EVENT = 'event'


class TripletCandidate(BaseModel):
    """One source-level subject-predicate-object decomposition."""

    subject: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object: str = Field(min_length=1)
    subject_kind: SemanticNodeKind
    object_kind: SemanticNodeKind


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
    triplets: list[TripletCandidate] = Field(default_factory=list)


class RawTriplet(_SemanticOccurrence):
    """Raw triplet occurrence connecting typed semantic components."""

    subject_uuid: str
    object_uuid: str
    predicate_uuid: str


class RawAssertion(BaseModel):
    """Complete raw assertion occurrence ready for graph persistence."""

    triplet: RawTriplet
    subject: SourceEntity | SourceEvent
    object: SourceEntity | SourceEvent
    predicate: SourcePredicate


__all__ = [
    'RawAssertion',
    'RawTriplet',
    'SemanticNodeKind',
    'TripletCandidate',
    'TripletDecompositionRequest',
    'TripletDecompositionResult',
]
