"""LangGraph state for global semantic consolidation."""

import operator
from typing import Annotated

from pydantic import BaseModel, Field

from kms2.core.model.global_semantic.global_entity_hub import (
    GlobalEntityHub,
    GlobalEntityHubCandidate,
    GlobalEntityHubJudgeResult,
    GlobalEntityHubMember,
    GlobalEntityHubRerankResult,
    GlobalEntityHubSynthesisResult,
)
from kms2.core.model.global_semantic.global_event_hub import (
    GlobalEventHub,
    GlobalEventHubCandidate,
    GlobalEventHubJudgeResult,
    GlobalEventHubMember,
    GlobalEventHubRerankResult,
    GlobalEventHubSynthesisResult,
)
from kms2.core.model.global_semantic.global_predicate_hub import (
    GlobalPredicateHub,
    GlobalPredicateHubCandidate,
    GlobalPredicateHubJudgeResult,
    GlobalPredicateHubMember,
    GlobalPredicateHubRerankResult,
    GlobalPredicateHubSynthesisResult,
)
from kms2.core.model.global_semantic.global_procedure_hub import (
    GlobalProcedureHub,
    GlobalProcedureHubCandidate,
    GlobalProcedureHubJudgeResult,
    GlobalProcedureHubMember,
    GlobalProcedureHubRerankResult,
    GlobalProcedureHubSynthesisResult,
)
from kms2.core.model.global_semantic.global_statement_hub import (
    GlobalStatementHub,
    GlobalStatementHubCandidate,
    GlobalStatementHubJudgeResult,
    GlobalStatementHubMember,
    GlobalStatementHubRerankResult,
    GlobalStatementHubSynthesisResult,
)
from kms2.core.model.global_semantic.global_triplet_hub import (
    GlobalTripletHub,
    GlobalTripletHubGroup,
    GlobalTripletHubSynthesisResult,
)


class GlobalSemanticState(BaseModel):
    """State shared by the global semantic hub phases."""

    global_entity_hub_candidates: list[GlobalEntityHubCandidate] = Field(
        default_factory=list
    )
    global_entity_hub_rerank_results: Annotated[
        list[GlobalEntityHubRerankResult], operator.add
    ] = Field(default_factory=list)
    global_entity_hub_direct_pairs: list[GlobalEntityHubCandidate] = Field(
        default_factory=list
    )
    global_entity_hub_borderline_pairs: list[GlobalEntityHubCandidate] = Field(
        default_factory=list
    )
    global_entity_hub_judge_results: Annotated[
        list[GlobalEntityHubJudgeResult], operator.add
    ] = Field(default_factory=list)
    global_entity_hub_accepted_pairs: list[GlobalEntityHubCandidate] = Field(
        default_factory=list
    )
    global_entity_hub_communities: list[list[GlobalEntityHubMember]] = Field(
        default_factory=list
    )
    global_entity_hub_synthesis_results: Annotated[
        list[GlobalEntityHubSynthesisResult], operator.add
    ] = Field(default_factory=list)
    global_entity_hub_synthesis_results_ordered: list[
        GlobalEntityHubSynthesisResult
    ] = Field(default_factory=list)
    global_entity_hubs: list[GlobalEntityHub] = Field(default_factory=list)
    global_entity_hub_memberships: list[list[str]] = Field(default_factory=list)
    global_entity_hub_count: int = 0

    global_event_hub_candidates: list[GlobalEventHubCandidate] = Field(
        default_factory=list
    )
    global_event_hub_rerank_results: Annotated[
        list[GlobalEventHubRerankResult], operator.add
    ] = Field(default_factory=list)
    global_event_hub_direct_pairs: list[GlobalEventHubCandidate] = Field(
        default_factory=list
    )
    global_event_hub_borderline_pairs: list[GlobalEventHubCandidate] = Field(
        default_factory=list
    )
    global_event_hub_judge_results: Annotated[
        list[GlobalEventHubJudgeResult], operator.add
    ] = Field(default_factory=list)
    global_event_hub_accepted_pairs: list[GlobalEventHubCandidate] = Field(
        default_factory=list
    )
    global_event_hub_communities: list[list[GlobalEventHubMember]] = Field(
        default_factory=list
    )
    global_event_hub_synthesis_results: Annotated[
        list[GlobalEventHubSynthesisResult], operator.add
    ] = Field(default_factory=list)
    global_event_hub_synthesis_results_ordered: list[
        GlobalEventHubSynthesisResult
    ] = Field(default_factory=list)
    global_event_hubs: list[GlobalEventHub] = Field(default_factory=list)
    global_event_hub_memberships: list[list[str]] = Field(default_factory=list)
    global_event_hub_count: int = 0

    global_predicate_hub_candidates: list[GlobalPredicateHubCandidate] = Field(
        default_factory=list
    )
    global_predicate_hub_rerank_results: Annotated[
        list[GlobalPredicateHubRerankResult], operator.add
    ] = Field(default_factory=list)
    global_predicate_hub_direct_pairs: list[GlobalPredicateHubCandidate] = (
        Field(default_factory=list)
    )
    global_predicate_hub_borderline_pairs: list[GlobalPredicateHubCandidate] = (
        Field(default_factory=list)
    )
    global_predicate_hub_judge_results: Annotated[
        list[GlobalPredicateHubJudgeResult], operator.add
    ] = Field(default_factory=list)
    global_predicate_hub_accepted_pairs: list[GlobalPredicateHubCandidate] = (
        Field(default_factory=list)
    )
    global_predicate_hub_communities: list[list[GlobalPredicateHubMember]] = (
        Field(default_factory=list)
    )
    global_predicate_hub_synthesis_results: Annotated[
        list[GlobalPredicateHubSynthesisResult], operator.add
    ] = Field(default_factory=list)
    global_predicate_hub_synthesis_results_ordered: list[
        GlobalPredicateHubSynthesisResult
    ] = Field(default_factory=list)
    global_predicate_hubs: list[GlobalPredicateHub] = Field(
        default_factory=list
    )

    global_predicate_hub_memberships: list[list[str]] = Field(
        default_factory=list
    )
    global_predicate_hub_count: int = 0
    global_triplet_count: int = 0
    global_triplet_hub_groups: list[GlobalTripletHubGroup] = Field(
        default_factory=list
    )
    global_triplet_hub_synthesis_results: Annotated[
        list[GlobalTripletHubSynthesisResult], operator.add
    ] = Field(default_factory=list)
    global_triplet_hub_synthesis_results_ordered: list[
        GlobalTripletHubSynthesisResult
    ] = Field(default_factory=list)
    global_triplet_hubs: list[GlobalTripletHub] = Field(default_factory=list)
    global_triplet_hub_memberships: list[list[str]] = Field(
        default_factory=list
    )
    global_triplet_hub_count: int = 0

    global_statement_hub_candidates: list[GlobalStatementHubCandidate] = Field(
        default_factory=list
    )
    global_statement_hub_rerank_results: Annotated[
        list[GlobalStatementHubRerankResult], operator.add
    ] = Field(default_factory=list)
    global_statement_hub_direct_pairs: list[GlobalStatementHubCandidate] = (
        Field(default_factory=list)
    )
    global_statement_hub_borderline_pairs: list[GlobalStatementHubCandidate] = (
        Field(default_factory=list)
    )
    global_statement_hub_judge_results: Annotated[
        list[GlobalStatementHubJudgeResult], operator.add
    ] = Field(default_factory=list)
    global_statement_hub_accepted_pairs: list[GlobalStatementHubCandidate] = (
        Field(default_factory=list)
    )
    global_statement_hub_communities: list[list[GlobalStatementHubMember]] = (
        Field(default_factory=list)
    )
    global_statement_hub_synthesis_results: Annotated[
        list[GlobalStatementHubSynthesisResult], operator.add
    ] = Field(default_factory=list)
    global_statement_hub_synthesis_results_ordered: list[
        GlobalStatementHubSynthesisResult
    ] = Field(default_factory=list)
    global_statement_hubs: list[GlobalStatementHub] = Field(
        default_factory=list
    )
    global_statement_hub_memberships: list[list[str]] = Field(
        default_factory=list
    )
    global_statement_hub_count: int = 0

    global_procedure_hub_candidates: list[GlobalProcedureHubCandidate] = Field(
        default_factory=list
    )
    global_procedure_hub_rerank_results: Annotated[
        list[GlobalProcedureHubRerankResult], operator.add
    ] = Field(default_factory=list)
    global_procedure_hub_direct_pairs: list[GlobalProcedureHubCandidate] = (
        Field(default_factory=list)
    )
    global_procedure_hub_borderline_pairs: list[GlobalProcedureHubCandidate] = (
        Field(default_factory=list)
    )
    global_procedure_hub_judge_results: Annotated[
        list[GlobalProcedureHubJudgeResult], operator.add
    ] = Field(default_factory=list)
    global_procedure_hub_accepted_pairs: list[GlobalProcedureHubCandidate] = (
        Field(default_factory=list)
    )
    global_procedure_hub_communities: list[list[GlobalProcedureHubMember]] = (
        Field(default_factory=list)
    )
    global_procedure_hub_synthesis_results: Annotated[
        list[GlobalProcedureHubSynthesisResult], operator.add
    ] = Field(default_factory=list)
    global_procedure_hub_synthesis_results_ordered: list[
        GlobalProcedureHubSynthesisResult
    ] = Field(default_factory=list)
    global_procedure_hubs: list[GlobalProcedureHub] = Field(
        default_factory=list
    )
    global_procedure_hub_memberships: list[list[str]] = Field(
        default_factory=list
    )
    global_procedure_hub_count: int = 0
