"""LangGraph registration for global triplet projection and hubs."""

from langgraph.graph import StateGraph

from kms2.node.global_semantic.global_triplet_hub import GlobalTripletHubNode
from kms2.node.global_semantic.global_triplet_hub_persistence import (
    GlobalTripletHubPersistenceNode,
)
from kms2.node.global_semantic.global_triplet_projection import (
    GlobalTripletProjectionNode,
)


def add_global_triplet_phase(
    graph: StateGraph,
    projection: GlobalTripletProjectionNode,
    hub: GlobalTripletHubNode,
    hub_persistence: GlobalTripletHubPersistenceNode,
) -> None:
    """Register global triplet projection, synthesis, embedding, and persistence."""
    graph.add_node('global_triplet_projection', projection.run)
    graph.add_node('global_triplet_hub_load', hub.load_groups)
    graph.add_node('global_triplet_hub_synthesis_worker', hub.synthesis_worker)
    graph.add_node(
        'global_triplet_hub_synthesis_collect', hub.collect_synthesis
    )
    graph.add_node('global_triplet_hub_embedding', hub.embed)
    graph.add_node('global_triplet_hub_persistence', hub_persistence.run)

    graph.add_edge('global_triplet_projection', 'global_triplet_hub_load')
    graph.add_conditional_edges(
        'global_triplet_hub_load',
        hub.dispatch_synthesis,
        [
            'global_triplet_hub_synthesis_worker',
            'global_triplet_hub_synthesis_collect',
        ],
    )
    graph.add_edge(
        'global_triplet_hub_synthesis_worker',
        'global_triplet_hub_synthesis_collect',
    )
    graph.add_edge(
        'global_triplet_hub_synthesis_collect', 'global_triplet_hub_embedding'
    )
    graph.add_edge(
        'global_triplet_hub_embedding', 'global_triplet_hub_persistence'
    )
