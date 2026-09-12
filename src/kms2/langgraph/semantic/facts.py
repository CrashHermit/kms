"""LangGraph registration for semantic fact extraction."""

from kms2.node.semantic.triplet import FactExtractionNode
from langgraph.graph import StateGraph


def add_fact_phase(graph: StateGraph, node: FactExtractionNode) -> None:
    """Register fact dispatch, workers, and collection."""
    graph.add_node('fact_worker', node.worker)
    graph.add_node('fact_collect', node.collect)
    graph.add_conditional_edges(
        'triplet_source_load',
        node.dispatch,
        ['fact_worker', 'fact_collect'],
    )
    graph.add_edge('fact_worker', 'fact_collect')
