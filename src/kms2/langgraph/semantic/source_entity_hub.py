"""LangGraph registration for source entity hub discovery."""

from kms2.node.semantic.source_entity_hub import SourceEntityHubNode
from kms2.node.semantic.source_entity_hub_persistence import (
    SourceEntityHubPersistenceNode,
)
from langgraph.graph import StateGraph


def add_source_entity_hub_phase(
    graph: StateGraph,
    entity: SourceEntityHubNode,
    persistence: SourceEntityHubPersistenceNode,
) -> None:
    """Register source entity hub discovery, synthesis, and persistence."""
    graph.add_node('source_entity_hub_load', entity.load_candidates)
    graph.add_node('source_entity_hub_rerank_worker', entity.rerank_worker)
    graph.add_node('source_entity_hub_rerank_collect', entity.collect_rerank)
    graph.add_node('source_entity_hub_judge_worker', entity.judge_worker)
    graph.add_node('source_entity_hub_judge_collect', entity.collect_judge)
    graph.add_node('source_entity_hub_communities', entity.detect_communities)
    graph.add_node(
        'source_entity_hub_synthesis_worker', entity.synthesis_worker
    )
    graph.add_node(
        'source_entity_hub_synthesis_collect', entity.collect_synthesis
    )
    graph.add_node('source_entity_hub_embedding', entity.embed)
    graph.add_node('source_entity_hub_persistence', persistence.run)

    graph.add_conditional_edges(
        'source_entity_hub_load',
        entity.dispatch_rerank,
        ['source_entity_hub_rerank_worker', 'source_entity_hub_rerank_collect'],
    )
    graph.add_edge(
        'source_entity_hub_rerank_worker', 'source_entity_hub_rerank_collect'
    )
    graph.add_conditional_edges(
        'source_entity_hub_rerank_collect',
        entity.dispatch_judge,
        ['source_entity_hub_judge_worker', 'source_entity_hub_judge_collect'],
    )
    graph.add_edge(
        'source_entity_hub_judge_worker', 'source_entity_hub_judge_collect'
    )
    graph.add_edge(
        'source_entity_hub_judge_collect', 'source_entity_hub_communities'
    )
    graph.add_conditional_edges(
        'source_entity_hub_communities',
        entity.dispatch_synthesis,
        [
            'source_entity_hub_synthesis_worker',
            'source_entity_hub_synthesis_collect',
        ],
    )
    graph.add_edge(
        'source_entity_hub_synthesis_worker',
        'source_entity_hub_synthesis_collect',
    )
    graph.add_edge(
        'source_entity_hub_synthesis_collect', 'source_entity_hub_embedding'
    )
    graph.add_edge(
        'source_entity_hub_embedding', 'source_entity_hub_persistence'
    )
