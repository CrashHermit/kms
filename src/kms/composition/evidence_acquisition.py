"""Composition for teaching evidence dispatch."""

from kms.core.media_budget import ImagePayloadBudget
from kms.node.evidence_acquisition.acquisition import EvidenceAcquisitionNode
from kms.node.teaching.tools import TeachingEvidenceTools
from kms.service.knowledge import KnowledgeService
from kms.service.knowledge_search import KnowledgeSearchService
from kms.service.learner_observation_search import (
    LearnerObservationSearchService,
)
from kms.service.learner_observations import LearnerObservationService
from kms.service.source_evidence import (
    SourceEvidenceService,
    SourceSearchService,
)
from kms.service.source_workspace import SourceWorkspaceService


def build_evidence_acquisition_node(
    source_workspace: SourceWorkspaceService,
    source_search: SourceSearchService,
    source_evidence: SourceEvidenceService,
    knowledge_search: KnowledgeSearchService,
    knowledge_service: KnowledgeService,
    *,
    learner_observation_search: LearnerObservationSearchService,
    learner_observations: LearnerObservationService,
    max_tool_calls: int,
    image_budget: ImagePayloadBudget,
    discovery_preview_characters: int,
    catalog_page_size: int,
) -> EvidenceAcquisitionNode:
    """Build direct evidence dispatch from invocation-owned services."""
    tools = TeachingEvidenceTools(
        source_search,
        source_evidence,
        source_workspace,
        knowledge_search,
        knowledge_service,
        learner_observation_search=learner_observation_search,
        learner_observations=learner_observations,
        image_budget=image_budget,
        discovery_preview_characters=discovery_preview_characters,
        catalog_page_size=catalog_page_size,
    )
    return EvidenceAcquisitionNode(
        tools,
        source_search,
        knowledge_search,
        max_tool_calls=max_tool_calls,
    )
