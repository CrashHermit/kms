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
    SourceEntityHub,
    SourceEventDescriptionRequest,
    SourceEventDescriptionResult,
    SourceEventHub,
    SourcePredicateDescriptionRequest,
    SourcePredicateDescriptionResult,
    SourcePredicateHub,
    SourceProcedureDescriptionRequest,
    SourceProcedureDescriptionResult,
    SourceProcedureHub,
    SourceStatementDescriptionRequest,
    SourceStatementDescriptionResult,
    SourceStatementHub,
    TripletDecompositionResult,
)


def _keep_loaded_blocks(
    current: list[SourceBlock], incoming: list[SourceBlock]
) -> list[SourceBlock]:
    """Keep one authoritative block snapshot when parallel loaders finish."""
    return current or incoming


class SemanticState(BaseModel):
    """State shared by semantic loading, extraction, and persistence."""

    source_uuid: str
    blocks: Annotated[list[SourceBlock], _keep_loaded_blocks] = Field(
        default_factory=list
    )
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
    source_entity_hub_count: int = 0
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
    source_event_hub_count: int = 0
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

    source_statement_description_requests: list[
        SourceStatementDescriptionRequest
    ] = Field(default_factory=list)
    source_statement_description_results: Annotated[
        list[SourceStatementDescriptionResult], operator.add
    ] = Field(default_factory=list)
    source_statement_embedding_results: Annotated[
        list[SourceStatementDescriptionResult], operator.add
    ] = Field(default_factory=list)
    source_statement_description_persisted_count: int = 0
    source_procedure_description_requests: list[
        SourceProcedureDescriptionRequest
    ] = Field(default_factory=list)
    source_procedure_description_results: Annotated[
        list[SourceProcedureDescriptionResult], operator.add
    ] = Field(default_factory=list)
    source_procedure_embedding_results: Annotated[
        list[SourceProcedureDescriptionResult], operator.add
    ] = Field(default_factory=list)
    source_procedure_description_persisted_count: int = 0
    source_predicate_hub_count: int = 0
    raw_assertions: list[RawAssertion] = Field(default_factory=list)
    source_entity_hubs: list[SourceEntityHub] = Field(default_factory=list)
    source_entity_hub_memberships: list[list[str]] = Field(default_factory=list)
    source_event_hubs: list[SourceEventHub] = Field(default_factory=list)
    source_event_hub_memberships: list[list[str]] = Field(default_factory=list)
    source_predicate_hubs: list[SourcePredicateHub] = Field(
        default_factory=list
    )
    source_predicate_hub_memberships: list[list[str]] = Field(
        default_factory=list
    )
    source_statement_hub_count: int = 0
    source_procedure_hub_count: int = 0
    source_statement_hubs: list[SourceStatementHub] = Field(
        default_factory=list
    )
    source_statement_hub_memberships: list[list[str]] = Field(
        default_factory=list
    )
    source_procedure_hubs: list[SourceProcedureHub] = Field(
        default_factory=list
    )
    source_procedure_hub_memberships: list[list[str]] = Field(
        default_factory=list
    )
