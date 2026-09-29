"""LangGraph registration for semantic triplet decomposition."""

from langgraph.graph import StateGraph

from kms2.node.source_semantic.source_triplet_decomposition import (
    SourceTripletDecompositionNode,
)
from kms2.node.source_semantic.source_triplet_fact_load import (
    SourceTripletFactLoadNode,
)


def add_source_triplet_phase(
    graph: StateGraph,
    fact_load: SourceTripletFactLoadNode,
    node: SourceTripletDecompositionNode,
) -> None:
    """Register persisted source-fact loading and source-triplet workers."""
    graph.add_node('source_triplet_fact_load', fact_load.run)
    graph.add_node('source_triplet_worker', node.worker)
    graph.add_node('source_triplet_collect', node.collect)
    graph.add_edge('source_fact_persistence', 'source_triplet_fact_load')
    graph.add_conditional_edges(
        'source_triplet_fact_load',
        node.dispatch,
        ['source_triplet_worker', 'source_triplet_collect'],
    )
    graph.add_edge('source_triplet_worker', 'source_triplet_collect')
