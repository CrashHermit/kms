"""LangGraph state for KMS2 semantic extraction."""

import operator
from typing import Annotated

from pydantic import BaseModel, Field

from kms2.core.model import (
    ExtractedFact,
    FactExtractionResult,
    SourceBlock,
    SourceEntityDescriptionRequest,
    SourceEntityDescriptionResult,
    SourceEntityHub,
    SourceEntityHubCandidate,
    SourceEntityHubJudgeResult,
    SourceEntityHubMember,
    SourceEntityHubRerankResult,
    SourceEntityHubSynthesisResult,
    SourceEventDescriptionRequest,
    SourceEventDescriptionResult,
    SourceEventHub,
    SourceEventHubCandidate,
    SourceEventHubJudgeResult,
    SourceEventHubMember,
    SourceEventHubRerankResult,
    SourceEventHubSynthesisResult,
    SourceFact,
    SourcePredicateDescriptionRequest,
    SourcePredicateDescriptionResult,
    SourcePredicateHub,
    SourcePredicateHubCandidate,
    SourcePredicateHubJudgeResult,
    SourcePredicateHubMember,
    SourcePredicateHubRerankResult,
    SourcePredicateHubSynthesisResult,
    SourceProcedureDescriptionRequest,
    SourceProcedureDescriptionResult,
    SourceProcedureHub,
    SourceProcedureHubCandidate,
    SourceProcedureHubJudgeResult,
    SourceProcedureHubMember,
    SourceProcedureHubRerankResult,
    SourceProcedureHubSynthesisResult,
    SourceStatementDescriptionRequest,
    SourceStatementDescriptionResult,
    SourceStatementHub,
    SourceStatementHubCandidate,
    SourceStatementHubJudgeResult,
    SourceStatementHubMember,
    SourceStatementHubRerankResult,
    SourceStatementHubSynthesisResult,
    SourceTripletHub,
    SourceTripletHubGroup,
    SourceTripletHubSynthesisResult,
    SourceTripletOccurrence,
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
    source_facts: list[SourceFact] = Field(default_factory=list)

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
    source_entity_hub_candidates: list[SourceEntityHubCandidate] = Field(
        default_factory=list
    )
    source_entity_hub_rerank_results: Annotated[
        list[SourceEntityHubRerankResult], operator.add
    ] = Field(default_factory=list)
    source_entity_hub_direct_pairs: list[SourceEntityHubCandidate] = Field(
        default_factory=list
    )
    source_entity_hub_borderline_pairs: list[SourceEntityHubCandidate] = Field(
        default_factory=list
    )
    source_entity_hub_judge_results: Annotated[
        list[SourceEntityHubJudgeResult], operator.add
    ] = Field(default_factory=list)
    source_entity_hub_accepted_pairs: list[SourceEntityHubCandidate] = Field(
        default_factory=list
    )
    source_entity_hub_communities: list[list[SourceEntityHubMember]] = Field(
        default_factory=list
    )
    source_entity_hub_synthesis_results: Annotated[
        list[SourceEntityHubSynthesisResult], operator.add
    ] = Field(default_factory=list)
    source_entity_hub_synthesis_results_ordered: list[
        SourceEntityHubSynthesisResult
    ] = Field(default_factory=list)
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
    source_event_hub_candidates: list[SourceEventHubCandidate] = Field(
        default_factory=list
    )
    source_event_hub_rerank_results: Annotated[
        list[SourceEventHubRerankResult], operator.add
    ] = Field(default_factory=list)
    source_event_hub_direct_pairs: list[SourceEventHubCandidate] = Field(
        default_factory=list
    )
    source_event_hub_borderline_pairs: list[SourceEventHubCandidate] = Field(
        default_factory=list
    )
    source_event_hub_judge_results: Annotated[
        list[SourceEventHubJudgeResult], operator.add
    ] = Field(default_factory=list)
    source_event_hub_accepted_pairs: list[SourceEventHubCandidate] = Field(
        default_factory=list
    )
    source_event_hub_communities: list[list[SourceEventHubMember]] = Field(
        default_factory=list
    )
    source_event_hub_synthesis_results: Annotated[
        list[SourceEventHubSynthesisResult], operator.add
    ] = Field(default_factory=list)
    source_event_hub_synthesis_results_ordered: list[
        SourceEventHubSynthesisResult
    ] = Field(default_factory=list)
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
    source_predicate_hub_candidates: list[SourcePredicateHubCandidate] = Field(
        default_factory=list
    )
    source_predicate_hub_rerank_results: Annotated[
        list[SourcePredicateHubRerankResult], operator.add
    ] = Field(default_factory=list)
    source_predicate_hub_direct_pairs: list[SourcePredicateHubCandidate] = (
        Field(default_factory=list)
    )
    source_predicate_hub_borderline_pairs: list[SourcePredicateHubCandidate] = (
        Field(default_factory=list)
    )
    source_predicate_hub_judge_results: Annotated[
        list[SourcePredicateHubJudgeResult], operator.add
    ] = Field(default_factory=list)
    source_predicate_hub_accepted_pairs: list[SourcePredicateHubCandidate] = (
        Field(default_factory=list)
    )
    source_predicate_hub_communities: list[list[SourcePredicateHubMember]] = (
        Field(default_factory=list)
    )
    source_predicate_hub_synthesis_results: Annotated[
        list[SourcePredicateHubSynthesisResult], operator.add
    ] = Field(default_factory=list)
    source_predicate_hub_synthesis_results_ordered: list[
        SourcePredicateHubSynthesisResult
    ] = Field(default_factory=list)
    source_predicate_description_persisted_count: int = 0
    source_predicate_hub_count: int = 0
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
    source_statement_hub_candidates: list[SourceStatementHubCandidate] = Field(
        default_factory=list
    )
    source_statement_hub_rerank_results: Annotated[
        list[SourceStatementHubRerankResult], operator.add
    ] = Field(default_factory=list)
    source_statement_hub_direct_pairs: list[SourceStatementHubCandidate] = (
        Field(default_factory=list)
    )
    source_statement_hub_borderline_pairs: list[SourceStatementHubCandidate] = (
        Field(default_factory=list)
    )
    source_statement_hub_judge_results: Annotated[
        list[SourceStatementHubJudgeResult], operator.add
    ] = Field(default_factory=list)
    source_statement_hub_accepted_pairs: list[SourceStatementHubCandidate] = (
        Field(default_factory=list)
    )
    source_statement_hub_communities: list[list[SourceStatementHubMember]] = (
        Field(default_factory=list)
    )
    source_statement_hub_synthesis_results: Annotated[
        list[SourceStatementHubSynthesisResult], operator.add
    ] = Field(default_factory=list)
    source_statement_hub_synthesis_results_ordered: list[
        SourceStatementHubSynthesisResult
    ] = Field(default_factory=list)
    source_statement_hub_count: int = 0
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
    source_procedure_hub_candidates: list[SourceProcedureHubCandidate] = Field(
        default_factory=list
    )
    source_procedure_hub_rerank_results: Annotated[
        list[SourceProcedureHubRerankResult], operator.add
    ] = Field(default_factory=list)
    source_procedure_hub_direct_pairs: list[SourceProcedureHubCandidate] = (
        Field(default_factory=list)
    )
    source_procedure_hub_borderline_pairs: list[SourceProcedureHubCandidate] = (
        Field(default_factory=list)
    )
    source_procedure_hub_judge_results: Annotated[
        list[SourceProcedureHubJudgeResult], operator.add
    ] = Field(default_factory=list)
    source_procedure_hub_accepted_pairs: list[SourceProcedureHubCandidate] = (
        Field(default_factory=list)
    )
    source_procedure_hub_communities: list[list[SourceProcedureHubMember]] = (
        Field(default_factory=list)
    )
    source_procedure_hub_synthesis_results: Annotated[
        list[SourceProcedureHubSynthesisResult], operator.add
    ] = Field(default_factory=list)
    source_procedure_hub_synthesis_results_ordered: list[
        SourceProcedureHubSynthesisResult
    ] = Field(default_factory=list)
    source_procedure_hub_count: int = 0

    triplet_occurrences: list[SourceTripletOccurrence] = Field(
        default_factory=list
    )
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
    source_triplet_hubs: list[SourceTripletHub] = Field(default_factory=list)
    source_triplet_hub_memberships: list[list[str]] = Field(
        default_factory=list
    )
    source_triplet_hub_count: int = 0
    source_triplet_hub_groups: list[SourceTripletHubGroup] = Field(
        default_factory=list
    )
    source_triplet_hub_synthesis_results: Annotated[
        list[SourceTripletHubSynthesisResult], operator.add
    ] = Field(default_factory=list)
    source_triplet_hub_synthesis_results_ordered: list[
        SourceTripletHubSynthesisResult
    ] = Field(default_factory=list)
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
