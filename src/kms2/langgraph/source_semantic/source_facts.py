"""LangGraph registration for semantic fact extraction."""

from langgraph.graph import StateGraph

from kms2.node.source_semantic.source_fact_extraction import (
    SourceFactExtractionNode,
)
from kms2.node.source_semantic.source_fact_persistence import (
    SourceFactPersistenceNode,
)


def add_source_fact_phase(
    graph: StateGraph,
    node: SourceFactExtractionNode,
    persistence: SourceFactPersistenceNode,
) -> None:
    """Register source-fact loading, workers, collection, and persistence."""
    graph.add_node('source_fact_worker', node.worker)
    graph.add_node('source_fact_collect', node.collect)
    graph.add_node('source_fact_persistence', persistence.run)
    graph.add_conditional_edges(
        'source_fact_load',
        node.dispatch,
        ['source_fact_worker', 'source_fact_collect'],
    )
    graph.add_edge('source_fact_worker', 'source_fact_collect')
    graph.add_edge('source_fact_collect', 'source_fact_persistence')
