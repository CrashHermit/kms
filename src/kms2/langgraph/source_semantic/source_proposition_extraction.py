"""LangGraph registration for proposition extraction and persistence."""

from langgraph.graph import StateGraph

from kms2.node.source_semantic.source_proposition_extraction import (
    SourcePropositionExtractionNode,
)
from kms2.node.source_semantic.source_proposition_persistence import (
    SourcePropositionPersistenceNode,
)


def add_source_proposition_extraction_phase(
    graph: StateGraph,
    extraction: SourcePropositionExtractionNode,
    persistence: SourcePropositionPersistenceNode,
) -> None:
    """Register proposition extraction workers and raw persistence."""
    graph.add_node('source_proposition_worker', extraction.worker)
    graph.add_node('source_proposition_collect', extraction.collect)
    graph.add_node('source_proposition_persistence', persistence.run)
    graph.add_conditional_edges(
        'source_semantic_load',
        extraction.dispatch_propositions,
        ['source_proposition_worker', 'source_proposition_collect'],
    )
    graph.add_edge('source_proposition_worker', 'source_proposition_collect')
    graph.add_edge(
        'source_proposition_collect', 'source_proposition_persistence'
    )
