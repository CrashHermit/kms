"""Semantic extraction, occurrence, description, and triplet models."""

from kms2.core.model.semantic.fact_extraction import (
    AtomicFact,
    ExtractedFact,
    FactExtractionInput,
    FactExtractionRequest,
    FactExtractionResult,
)
from kms2.core.model.semantic.source_entity import (
    SourceEntity,
    SourceEntityDescriptionRequest,
    SourceEntityDescriptionResult,
    SourceEntityDescriptionTarget,
)
from kms2.core.model.semantic.source_event import (
    SourceEvent,
    SourceEventDescriptionRequest,
    SourceEventDescriptionResult,
    SourceEventDescriptionTarget,
)
from kms2.core.model.semantic.source_predicate import (
    SourcePredicate,
    SourcePredicateDescriptionRequest,
    SourcePredicateDescriptionResult,
    SourcePredicateDescriptionTarget,
)
from kms2.core.model.semantic.term_description import TermDescriptionInput
from kms2.core.model.semantic.triplet import (
    RawAssertion,
    RawTriplet,
    SemanticNodeKind,
    TripletCandidate,
    TripletDecompositionRequest,
    TripletDecompositionResult,
)

__all__ = [
    'AtomicFact',
    'ExtractedFact',
    'FactExtractionInput',
    'FactExtractionRequest',
    'FactExtractionResult',
    'RawAssertion',
    'RawTriplet',
    'SemanticNodeKind',
    'SourceEntity',
    'SourceEntityDescriptionRequest',
    'SourceEntityDescriptionResult',
    'SourceEntityDescriptionTarget',
    'SourceEvent',
    'SourceEventDescriptionRequest',
    'SourceEventDescriptionResult',
    'SourceEventDescriptionTarget',
    'SourcePredicate',
    'SourcePredicateDescriptionRequest',
    'SourcePredicateDescriptionResult',
    'SourcePredicateDescriptionTarget',
    'TermDescriptionInput',
    'TripletCandidate',
    'TripletDecompositionRequest',
    'TripletDecompositionResult',
]
