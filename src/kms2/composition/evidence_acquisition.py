"""Composition for the isolated teaching evidence action graph."""

from kms2.langgraph.evidence_acquisition.graph import EvidenceAcquisitionGraph
from kms2.langgraph.research.graph import ResearchGraph
from kms2.langgraph.source_admission.graph import SourceAdmissionGraph
from kms2.node.evidence_acquisition.acquisition import EvidenceAcquisitionNode
from kms2.node.teaching.tools import TeachingEvidenceTools
from kms2.service.knowledge import KnowledgeService
from kms2.service.knowledge_search import KnowledgeSearchService
from kms2.service.source_evidence import (
    SourceEvidenceService,
    SourceSearchService,
)


def build_evidence_acquisition_graph(
    source_search: SourceSearchService,
    source_evidence: SourceEvidenceService,
    knowledge_search: KnowledgeSearchService,
    knowledge_service: KnowledgeService,
    *,
    researcher: ResearchGraph | None,
    source_admission: SourceAdmissionGraph | None,
    discovery_preview_characters: int,
    catalog_page_size: int,
) -> EvidenceAcquisitionGraph:
    """Build one workspace action graph from invocation-owned services."""
    tools = TeachingEvidenceTools(
        source_search,
        source_evidence,
        knowledge_search,
        knowledge_service,
        discovery_preview_characters=discovery_preview_characters,
        catalog_page_size=catalog_page_size,
    )
    node = EvidenceAcquisitionNode(
        tools,
        source_search,
        knowledge_search,
        researcher,
        source_admission,
        discovery_preview_characters=discovery_preview_characters,
    )
    return EvidenceAcquisitionGraph(node)
