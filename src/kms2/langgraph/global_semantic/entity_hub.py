"""LangGraph registration for global entity hub discovery."""

from kms2.node.global_semantic.entity_hub import GlobalEntityHubNode
from kms2.node.global_semantic.entity_hub_persistence import (
    GlobalEntityHubPersistenceNode,
)
from langgraph.graph import StateGraph


def add_global_entity_hub_phase(
    graph: StateGraph,
    entity: GlobalEntityHubNode,
    persistence: GlobalEntityHubPersistenceNode,
) -> None:
    """Register global entity hub discovery and persistence."""
    graph.add_node('global_entity_hub_load', entity.load_candidates)
    graph.add_node('global_entity_hub_rerank_worker', entity.rerank_worker)
    graph.add_node('global_entity_hub_rerank_collect', entity.collect_rerank)
    graph.add_node('global_entity_hub_judge_worker', entity.judge_worker)
    graph.add_node('global_entity_hub_judge_collect', entity.collect_judge)
    graph.add_node('global_entity_hub_communities', entity.detect_communities)
    graph.add_node(
        'global_entity_hub_synthesis_worker', entity.synthesis_worker
    )
    graph.add_node(
        'global_entity_hub_synthesis_collect', entity.collect_synthesis
    )
    graph.add_node('global_entity_hub_embedding', entity.embed)
    graph.add_node('global_entity_hub_persistence', persistence.run)

    graph.add_conditional_edges(
        'global_entity_hub_load',
        entity.dispatch_rerank,
        [
            'global_entity_hub_rerank_worker',
            'global_entity_hub_rerank_collect',
        ],
    )
    graph.add_edge(
        'global_entity_hub_rerank_worker',
        'global_entity_hub_rerank_collect',
    )
    graph.add_conditional_edges(
        'global_entity_hub_rerank_collect',
        entity.dispatch_judge,
        [
            'global_entity_hub_judge_worker',
            'global_entity_hub_judge_collect',
        ],
    )
    graph.add_edge(
        'global_entity_hub_judge_worker',
        'global_entity_hub_judge_collect',
    )
    graph.add_edge(
        'global_entity_hub_judge_collect',
        'global_entity_hub_communities',
    )
    graph.add_conditional_edges(
        'global_entity_hub_communities',
        entity.dispatch_synthesis,
        [
            'global_entity_hub_synthesis_worker',
            'global_entity_hub_synthesis_collect',
        ],
    )
    graph.add_edge(
        'global_entity_hub_synthesis_worker',
        'global_entity_hub_synthesis_collect',
    )
    graph.add_edge(
        'global_entity_hub_synthesis_collect',
        'global_entity_hub_embedding',
    )
    graph.add_edge(
        'global_entity_hub_embedding', 'global_entity_hub_persistence'
    )


__all__ = ['add_global_entity_hub_phase']
