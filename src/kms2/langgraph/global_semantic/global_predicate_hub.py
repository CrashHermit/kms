"""LangGraph registration for global predicate hub discovery."""

from langgraph.graph import StateGraph

from kms2.node.global_semantic.global_predicate_hub import (
    GlobalPredicateHubNode,
)
from kms2.node.global_semantic.global_predicate_hub_persistence import (
    GlobalPredicateHubPersistenceNode,
)


def add_global_predicate_hub_phase(
    graph: StateGraph,
    predicate: GlobalPredicateHubNode,
    persistence: GlobalPredicateHubPersistenceNode,
) -> None:
    """Register global predicate hub discovery and persistence."""
    graph.add_node('global_predicate_hub_load', predicate.load_candidates)
    graph.add_node(
        'global_predicate_hub_rerank_worker', predicate.rerank_worker
    )
    graph.add_node(
        'global_predicate_hub_rerank_collect', predicate.collect_rerank
    )
    graph.add_node('global_predicate_hub_judge_worker', predicate.judge_worker)
    graph.add_node(
        'global_predicate_hub_judge_collect', predicate.collect_judge
    )
    graph.add_node(
        'global_predicate_hub_communities', predicate.detect_communities
    )
    graph.add_node(
        'global_predicate_hub_synthesis_worker', predicate.synthesis_worker
    )
    graph.add_node(
        'global_predicate_hub_synthesis_collect', predicate.collect_synthesis
    )
    graph.add_node('global_predicate_hub_embedding', predicate.embed)
    graph.add_node('global_predicate_hub_persistence', persistence.run)

    graph.add_conditional_edges(
        'global_predicate_hub_load',
        predicate.dispatch_rerank,
        [
            'global_predicate_hub_rerank_worker',
            'global_predicate_hub_rerank_collect',
        ],
    )
    graph.add_edge(
        'global_predicate_hub_rerank_worker',
        'global_predicate_hub_rerank_collect',
    )
    graph.add_conditional_edges(
        'global_predicate_hub_rerank_collect',
        predicate.dispatch_judge,
        [
            'global_predicate_hub_judge_worker',
            'global_predicate_hub_judge_collect',
        ],
    )
    graph.add_edge(
        'global_predicate_hub_judge_worker',
        'global_predicate_hub_judge_collect',
    )
    graph.add_edge(
        'global_predicate_hub_judge_collect',
        'global_predicate_hub_communities',
    )
    graph.add_conditional_edges(
        'global_predicate_hub_communities',
        predicate.dispatch_synthesis,
        [
            'global_predicate_hub_synthesis_worker',
            'global_predicate_hub_synthesis_collect',
        ],
    )
    graph.add_edge(
        'global_predicate_hub_synthesis_worker',
        'global_predicate_hub_synthesis_collect',
    )
    graph.add_edge(
        'global_predicate_hub_synthesis_collect',
        'global_predicate_hub_embedding',
    )
    graph.add_edge(
        'global_predicate_hub_embedding', 'global_predicate_hub_persistence'
    )
