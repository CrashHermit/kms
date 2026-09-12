"""LangGraph state for KMS2 semantic extraction."""

import operator
from typing import Annotated

from pydantic import BaseModel, Field

from kms2.core.model import (
    EntityEnrichmentRequest,
    EntityEnrichmentResult,
    EventEnrichmentRequest,
    EventEnrichmentResult,
    ExtractedFact,
    FactExtractionResult,
    PredicateEnrichmentRequest,
    PredicateEnrichmentResult,
    RawAssertion,
    SourceBlock,
    TripletDecompositionResult,
)


class SemanticState(BaseModel):
    """State shared by semantic loading, extraction, and persistence."""

    source_uuid: str
    blocks: list[SourceBlock] = Field(default_factory=list)
    fact_results: Annotated[list[FactExtractionResult], operator.add] = Field(
        default_factory=list
    )
    extracted_facts: list[ExtractedFact] = Field(default_factory=list)
    triplet_results: Annotated[
        list[TripletDecompositionResult], operator.add
    ] = Field(default_factory=list)

    entity_requests: list[EntityEnrichmentRequest] = Field(default_factory=list)
    entity_description_results: Annotated[
        list[EntityEnrichmentResult], operator.add
    ] = Field(default_factory=list)
    entity_embedding_results: Annotated[
        list[EntityEnrichmentResult], operator.add
    ] = Field(default_factory=list)
    entity_persisted_count: int = 0
    event_requests: list[EventEnrichmentRequest] = Field(default_factory=list)
    event_description_results: Annotated[
        list[EventEnrichmentResult], operator.add
    ] = Field(default_factory=list)
    event_embedding_results: Annotated[
        list[EventEnrichmentResult], operator.add
    ] = Field(default_factory=list)
    event_persisted_count: int = 0
    predicate_requests: list[PredicateEnrichmentRequest] = Field(
        default_factory=list
    )
    predicate_description_results: Annotated[
        list[PredicateEnrichmentResult], operator.add
    ] = Field(default_factory=list)
    predicate_embedding_results: Annotated[
        list[PredicateEnrichmentResult], operator.add
    ] = Field(default_factory=list)
    predicate_persisted_count: int = 0
    raw_assertions: list[RawAssertion] = Field(default_factory=list)
