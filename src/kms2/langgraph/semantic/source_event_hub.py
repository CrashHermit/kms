"""LangGraph registration for source event hub discovery."""

from kms2.node.semantic.source_event_hub import SourceEventHubNode
from kms2.node.semantic.source_event_hub_persistence import (
    SourceEventHubPersistenceNode,
)
from langgraph.graph import StateGraph


def add_source_event_hub_phase(
    graph: StateGraph,
    event: SourceEventHubNode,
    persistence: SourceEventHubPersistenceNode,
) -> None:
    """Register source event hub discovery, synthesis, and persistence."""
    graph.add_node('source_event_hub_load', event.load_candidates)
    graph.add_node('source_event_hub_rerank_worker', event.rerank_worker)
    graph.add_node('source_event_hub_rerank_collect', event.collect_rerank)
    graph.add_node('source_event_hub_judge_worker', event.judge_worker)
    graph.add_node('source_event_hub_judge_collect', event.collect_judge)
    graph.add_node('source_event_hub_communities', event.detect_communities)
    graph.add_node('source_event_hub_synthesis_worker', event.synthesis_worker)
    graph.add_node(
        'source_event_hub_synthesis_collect', event.collect_synthesis
    )
    graph.add_node('source_event_hub_embedding', event.embed)
    graph.add_node('source_event_hub_persistence', persistence.run)

    graph.add_conditional_edges(
        'source_event_hub_load',
        event.dispatch_rerank,
        ['source_event_hub_rerank_worker', 'source_event_hub_rerank_collect'],
    )
    graph.add_edge(
        'source_event_hub_rerank_worker', 'source_event_hub_rerank_collect'
    )
    graph.add_conditional_edges(
        'source_event_hub_rerank_collect',
        event.dispatch_judge,
        ['source_event_hub_judge_worker', 'source_event_hub_judge_collect'],
    )
    graph.add_edge(
        'source_event_hub_judge_worker', 'source_event_hub_judge_collect'
    )
    graph.add_edge(
        'source_event_hub_judge_collect', 'source_event_hub_communities'
    )
    graph.add_conditional_edges(
        'source_event_hub_communities',
        event.dispatch_synthesis,
        [
            'source_event_hub_synthesis_worker',
            'source_event_hub_synthesis_collect',
        ],
    )
    graph.add_edge(
        'source_event_hub_synthesis_worker',
        'source_event_hub_synthesis_collect',
    )
    graph.add_edge(
        'source_event_hub_synthesis_collect', 'source_event_hub_embedding'
    )
    graph.add_edge('source_event_hub_embedding', 'source_event_hub_persistence')
