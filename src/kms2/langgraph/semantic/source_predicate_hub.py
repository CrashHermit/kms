"""LangGraph registration for source predicate hub discovery."""

from kms2.node.semantic.source_predicate_hub import SourcePredicateHubNode
from kms2.node.semantic.source_predicate_hub_persistence import (
    SourcePredicateHubPersistenceNode,
)
from langgraph.graph import StateGraph


def add_source_predicate_hub_phase(
    graph: StateGraph,
    predicate: SourcePredicateHubNode,
    persistence: SourcePredicateHubPersistenceNode,
) -> None:
    """Register source predicate hub discovery, synthesis, and persistence."""
    graph.add_node('source_predicate_hub_load', predicate.load_candidates)
    graph.add_node(
        'source_predicate_hub_rerank_worker', predicate.rerank_worker
    )
    graph.add_node(
        'source_predicate_hub_rerank_collect', predicate.collect_rerank
    )
    graph.add_node('source_predicate_hub_judge_worker', predicate.judge_worker)
    graph.add_node(
        'source_predicate_hub_judge_collect', predicate.collect_judge
    )
    graph.add_node(
        'source_predicate_hub_communities', predicate.detect_communities
    )
    graph.add_node(
        'source_predicate_hub_synthesis_worker', predicate.synthesis_worker
    )
    graph.add_node(
        'source_predicate_hub_synthesis_collect', predicate.collect_synthesis
    )
    graph.add_node('source_predicate_hub_embedding', predicate.embed)
    graph.add_node('source_predicate_hub_persistence', persistence.run)

    graph.add_conditional_edges(
        'source_predicate_hub_load',
        predicate.dispatch_rerank,
        [
            'source_predicate_hub_rerank_worker',
            'source_predicate_hub_rerank_collect',
        ],
    )
    graph.add_edge(
        'source_predicate_hub_rerank_worker',
        'source_predicate_hub_rerank_collect',
    )
    graph.add_conditional_edges(
        'source_predicate_hub_rerank_collect',
        predicate.dispatch_judge,
        [
            'source_predicate_hub_judge_worker',
            'source_predicate_hub_judge_collect',
        ],
    )
    graph.add_edge(
        'source_predicate_hub_judge_worker',
        'source_predicate_hub_judge_collect',
    )
    graph.add_edge(
        'source_predicate_hub_judge_collect',
        'source_predicate_hub_communities',
    )
    graph.add_conditional_edges(
        'source_predicate_hub_communities',
        predicate.dispatch_synthesis,
        [
            'source_predicate_hub_synthesis_worker',
            'source_predicate_hub_synthesis_collect',
        ],
    )
    graph.add_edge(
        'source_predicate_hub_synthesis_worker',
        'source_predicate_hub_synthesis_collect',
    )
    graph.add_edge(
        'source_predicate_hub_synthesis_collect',
        'source_predicate_hub_embedding',
    )
    graph.add_edge(
        'source_predicate_hub_embedding', 'source_predicate_hub_persistence'
    )
