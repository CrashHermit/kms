"""LangGraph registration for source triplet hub synthesis."""

from kms2.node.semantic.source_triplet_hub import SourceTripletHubNode
from kms2.node.semantic.source_triplet_hub_persistence import (
    SourceTripletHubPersistenceNode,
)
from langgraph.graph import StateGraph


def add_source_triplet_hub_phase(
    graph: StateGraph,
    triplet: SourceTripletHubNode,
    persistence: SourceTripletHubPersistenceNode,
) -> None:
    """Register source triplet hub synthesis, embedding, and persistence."""
    graph.add_node('source_triplet_hub_load', triplet.load_groups)
    graph.add_node(
        'source_triplet_hub_synthesis_worker', triplet.synthesis_worker
    )
    graph.add_node(
        'source_triplet_hub_synthesis_collect', triplet.collect_synthesis
    )
    graph.add_node('source_triplet_hub_embedding', triplet.embed)
    graph.add_node('source_triplet_hub_persistence', persistence.run)

    graph.add_conditional_edges(
        'source_triplet_hub_load',
        triplet.dispatch_synthesis,
        [
            'source_triplet_hub_synthesis_worker',
            'source_triplet_hub_synthesis_collect',
        ],
    )
    graph.add_edge(
        'source_triplet_hub_synthesis_worker',
        'source_triplet_hub_synthesis_collect',
    )
    graph.add_edge(
        'source_triplet_hub_synthesis_collect',
        'source_triplet_hub_embedding',
    )
    graph.add_edge(
        'source_triplet_hub_embedding', 'source_triplet_hub_persistence'
    )
