"""LangGraph registration for global event hub discovery."""

from kms2.node.global_semantic.event_hub import GlobalEventHubNode
from kms2.node.global_semantic.event_hub_persistence import (
    GlobalEventHubPersistenceNode,
)
from langgraph.graph import StateGraph


def add_global_event_hub_phase(
    graph: StateGraph,
    event: GlobalEventHubNode,
    persistence: GlobalEventHubPersistenceNode,
) -> None:
    """Register global event hub discovery and persistence."""
    graph.add_node('global_event_hub_load', event.load_candidates)
    graph.add_node('global_event_hub_rerank_worker', event.rerank_worker)
    graph.add_node('global_event_hub_rerank_collect', event.collect_rerank)
    graph.add_node('global_event_hub_judge_worker', event.judge_worker)
    graph.add_node('global_event_hub_judge_collect', event.collect_judge)
    graph.add_node('global_event_hub_communities', event.detect_communities)
    graph.add_node('global_event_hub_synthesis_worker', event.synthesis_worker)
    graph.add_node(
        'global_event_hub_synthesis_collect', event.collect_synthesis
    )
    graph.add_node('global_event_hub_embedding', event.embed)
    graph.add_node('global_event_hub_persistence', persistence.run)

    graph.add_conditional_edges(
        'global_event_hub_load',
        event.dispatch_rerank,
        [
            'global_event_hub_rerank_worker',
            'global_event_hub_rerank_collect',
        ],
    )
    graph.add_edge(
        'global_event_hub_rerank_worker',
        'global_event_hub_rerank_collect',
    )
    graph.add_conditional_edges(
        'global_event_hub_rerank_collect',
        event.dispatch_judge,
        [
            'global_event_hub_judge_worker',
            'global_event_hub_judge_collect',
        ],
    )
    graph.add_edge(
        'global_event_hub_judge_worker',
        'global_event_hub_judge_collect',
    )
    graph.add_edge(
        'global_event_hub_judge_collect',
        'global_event_hub_communities',
    )
    graph.add_conditional_edges(
        'global_event_hub_communities',
        event.dispatch_synthesis,
        [
            'global_event_hub_synthesis_worker',
            'global_event_hub_synthesis_collect',
        ],
    )
    graph.add_edge(
        'global_event_hub_synthesis_worker',
        'global_event_hub_synthesis_collect',
    )
    graph.add_edge(
        'global_event_hub_synthesis_collect',
        'global_event_hub_embedding',
    )
    graph.add_edge('global_event_hub_embedding', 'global_event_hub_persistence')


__all__ = ['add_global_event_hub_phase']
