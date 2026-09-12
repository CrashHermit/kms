"""Source-faithful semantic extraction and assertion models."""

from enum import StrEnum

from pydantic import BaseModel, Field

from kms2.core.model.base import Vertex
from kms2.core.model.context import SourceBlockContext, SourceContextWindow
from kms2.core.model.source import SourceBlock


class SemanticNodeKind(StrEnum):
    """Endpoint kinds emitted by source-level triplet decomposition."""

    ENTITY = 'entity'
    EVENT = 'event'


class FactExtractionInput(BaseModel):
    """UUID-free context supplied to the fact extractor."""

    context_before: list[SourceBlockContext] = Field(default_factory=list)
    target_block: SourceBlockContext
    context_after: list[SourceBlockContext] = Field(default_factory=list)


class FactExtractionRequest(BaseModel):
    """One canonical target block with its model-facing context."""

    source_uuid: str
    target_block: SourceBlock
    window: SourceContextWindow

    def model_input(self) -> FactExtractionInput:
        """Return the fact model's UUID-free canonical boundary."""
        return FactExtractionInput(
            context_before=self.window.context_before,
            target_block=self.window.target[0],
            context_after=self.window.context_after,
        )


class AtomicFact(BaseModel):
    """One source-faithful atomic assertion returned by the first pass."""

    text: str = Field(min_length=1)


class ExtractedFact(BaseModel):
    """One atomic fact reattached to its authoritative source block."""

    source_uuid: str
    source_block_uuid: str
    text: str = Field(min_length=1)


class FactExtractionResult(BaseModel):
    """Facts extracted from one source block."""

    target_block_uuid: str
    facts: list[ExtractedFact] = Field(default_factory=list)


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


class _SemanticOccurrence(Vertex):
    """Common provenance carried by one raw semantic occurrence."""

    source_uuid: str
    source_block_uuid: str


class Entity(_SemanticOccurrence):
    """Raw entity occurrence attached to one source assertion."""

    name: str = Field(min_length=1)


class Event(_SemanticOccurrence):
    """Raw event occurrence attached to one source assertion."""

    name: str = Field(min_length=1)


class Predicate(_SemanticOccurrence):
    """Raw predicate occurrence attached to one source assertion."""

    predicate: str = Field(min_length=1)


class TermEnrichmentInput(BaseModel):
    """One typed occurrence and its source-local model context."""

    term: str = Field(min_length=1)
    context_before: list[SourceBlockContext] = Field(default_factory=list)
    target_block: SourceBlockContext
    context_after: list[SourceBlockContext] = Field(default_factory=list)


class EntityEnrichmentTarget(Entity):
    """Raw entity occurrence selected for typed enrichment."""


class EventEnrichmentTarget(Event):
    """Raw event occurrence selected for typed enrichment."""


class PredicateEnrichmentTarget(Predicate):
    """Raw predicate occurrence selected for typed enrichment."""


class EntityEnrichmentRequest(BaseModel):
    """One entity occurrence and its enrichment input."""

    target: EntityEnrichmentTarget
    model_input: TermEnrichmentInput


class EventEnrichmentRequest(BaseModel):
    """One event occurrence and its enrichment input."""

    target: EventEnrichmentTarget
    model_input: TermEnrichmentInput


class PredicateEnrichmentRequest(BaseModel):
    """One predicate occurrence and its enrichment input."""

    target: PredicateEnrichmentTarget
    model_input: TermEnrichmentInput


class EntityEnrichmentResult(EntityEnrichmentTarget):
    """One enriched entity occurrence before graph persistence."""

    description: str = Field(min_length=1)
    embedding: list[float] | None = None


class EventEnrichmentResult(EventEnrichmentTarget):
    """One enriched event occurrence before graph persistence."""

    description: str = Field(min_length=1)
    embedding: list[float] | None = None


class PredicateEnrichmentResult(PredicateEnrichmentTarget):
    """One enriched predicate occurrence before graph persistence."""

    description: str = Field(min_length=1)
    embedding: list[float] | None = None


class RawTriplet(_SemanticOccurrence):
    """Raw triplet occurrence connecting typed semantic components."""

    subject_uuid: str
    object_uuid: str
    predicate_uuid: str


class RawAssertion(BaseModel):
    """Complete raw assertion occurrence ready for graph persistence."""

    triplet: RawTriplet
    subject: Entity | Event
    object: Entity | Event
    predicate: Predicate


__all__ = [
    'AtomicFact',
    'Entity',
    'EntityEnrichmentRequest',
    'EntityEnrichmentResult',
    'EntityEnrichmentTarget',
    'Event',
    'EventEnrichmentRequest',
    'EventEnrichmentResult',
    'EventEnrichmentTarget',
    'ExtractedFact',
    'FactExtractionInput',
    'FactExtractionRequest',
    'FactExtractionResult',
    'Predicate',
    'PredicateEnrichmentRequest',
    'PredicateEnrichmentResult',
    'PredicateEnrichmentTarget',
    'RawAssertion',
    'RawTriplet',
    'SemanticNodeKind',
    'TermEnrichmentInput',
    'TripletCandidate',
    'TripletDecompositionRequest',
    'TripletDecompositionResult',
]
