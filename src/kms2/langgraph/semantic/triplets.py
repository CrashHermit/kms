"""LangGraph registration for semantic triplet decomposition."""

from kms2.node.semantic.triplet import TripletDecompositionNode
from langgraph.graph import StateGraph


def add_triplet_phase(
    graph: StateGraph, node: TripletDecompositionNode
) -> None:
    """Register triplet dispatch, workers, and collection."""
    graph.add_node('triplet_worker', node.worker)
    graph.add_node('triplet_collect', node.collect)
    graph.add_conditional_edges(
        'fact_collect',
        node.dispatch,
        ['triplet_worker', 'triplet_collect'],
    )
    graph.add_edge('triplet_worker', 'triplet_collect')
