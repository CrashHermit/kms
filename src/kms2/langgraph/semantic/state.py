"""LangGraph state for KMS2 semantic extraction."""

import operator
from typing import Annotated

from pydantic import BaseModel, Field

from kms2.core.model import (
    ExtractedFact,
    FactExtractionResult,
    RawAssertion,
    SourceBlock,
    SourceEntityDescriptionRequest,
    SourceEntityDescriptionResult,
    SourceEventDescriptionRequest,
    SourceEventDescriptionResult,
    SourcePredicateDescriptionRequest,
    SourcePredicateDescriptionResult,
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

    source_entity_description_requests: list[SourceEntityDescriptionRequest] = (
        Field(default_factory=list)
    )
    source_entity_description_results: Annotated[
        list[SourceEntityDescriptionResult], operator.add
    ] = Field(default_factory=list)
    source_entity_embedding_results: Annotated[
        list[SourceEntityDescriptionResult], operator.add
    ] = Field(default_factory=list)
    source_entity_description_persisted_count: int = 0
    source_event_description_requests: list[SourceEventDescriptionRequest] = (
        Field(default_factory=list)
    )
    source_event_description_results: Annotated[
        list[SourceEventDescriptionResult], operator.add
    ] = Field(default_factory=list)
    source_event_embedding_results: Annotated[
        list[SourceEventDescriptionResult], operator.add
    ] = Field(default_factory=list)
    source_event_description_persisted_count: int = 0
    source_predicate_description_requests: list[
        SourcePredicateDescriptionRequest
    ] = Field(default_factory=list)
    source_predicate_description_results: Annotated[
        list[SourcePredicateDescriptionResult], operator.add
    ] = Field(default_factory=list)
    source_predicate_embedding_results: Annotated[
        list[SourcePredicateDescriptionResult], operator.add
    ] = Field(default_factory=list)
    source_predicate_description_persisted_count: int = 0
    raw_assertions: list[RawAssertion] = Field(default_factory=list)
