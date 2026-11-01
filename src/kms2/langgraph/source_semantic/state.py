"""LangGraph state for KMS2 semantic extraction."""

import operator
from typing import Annotated

from pydantic import BaseModel, Field

from kms2.core.model.block import SourceBlock
from kms2.core.model.source_semantic.source_entity import (
    SourceEntity,
    SourceEntityDescriptionRequest,
    SourceEntityDescriptionResult,
)
from kms2.core.model.source_semantic.source_entity_extraction import (
    SourceEntityExtractionResult,
)
from kms2.core.model.source_semantic.source_entity_hub import (
    SourceEntityHub,
    SourceEntityHubCandidate,
    SourceEntityHubJudgeResult,
    SourceEntityHubMember,
    SourceEntityHubRerankResult,
)
from kms2.core.model.source_semantic.source_procedure_hub import (
    SourceProcedureHub,
    SourceProcedureHubCandidate,
    SourceProcedureHubJudgeResult,
    SourceProcedureHubMember,
    SourceProcedureHubRerankResult,
    SourceProcedureHubSynthesisResult,
)
from kms2.core.model.source_semantic.source_statement_hub import (
    SourceStatementHub,
    SourceStatementHubCandidate,
    SourceStatementHubJudgeResult,
    SourceStatementHubMember,
    SourceStatementHubRerankResult,
    SourceStatementHubSynthesisResult,
)


class SourceSemanticState(BaseModel):
    """State shared by semantic loading, extraction, and persistence."""

    source_uuid: str
    blocks: list[SourceBlock] = Field(default_factory=list)
    source_entity_extraction_results: Annotated[
        list[SourceEntityExtractionResult], operator.add
    ] = Field(default_factory=list)
    source_entities: list[SourceEntity] = Field(default_factory=list)
    source_entity_persisted_count: int = 0
    source_entity_description_requests: list[SourceEntityDescriptionRequest] = (
        Field(default_factory=list)
    )
    source_entity_description_results: Annotated[
        list[SourceEntityDescriptionResult], operator.add
    ] = Field(default_factory=list)
    source_entity_embedding_results: list[SourceEntityDescriptionResult] = (
        Field(default_factory=list)
    )
    source_entity_description_persisted_count: int = 0
    source_entity_hub_candidates: list[SourceEntityHubCandidate] = Field(
        default_factory=list
    )
    source_entity_hub_rerank_results: Annotated[
        list[SourceEntityHubRerankResult], operator.add
    ] = Field(default_factory=list)
    source_entity_hub_reranked_candidates: list[SourceEntityHubCandidate] = (
        Field(default_factory=list)
    )
    source_entity_hub_judge_results: Annotated[
        list[SourceEntityHubJudgeResult], operator.add
    ] = Field(default_factory=list)
    source_entity_hub_groups: list[list[SourceEntityHubMember]] = Field(
        default_factory=list
    )
    source_entity_hub_members: list[SourceEntityHubMember] = Field(
        default_factory=list
    )
    source_entity_hub_decision_cache: dict[str, bool] = Field(
        default_factory=dict
    )
    source_entity_hubs: list[SourceEntityHub] = Field(default_factory=list)
    source_entity_hub_memberships: list[list[str]] = Field(default_factory=list)
    source_entity_hub_count: int = 0
    source_entity_hub_round_merged: bool = False
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
