"""LangGraph registration for source proposition hub discovery."""

from langgraph.graph import StateGraph

from kms2.node.source_semantic.source_proposition_hub import (
    SourcePropositionHubNode,
)
from kms2.node.source_semantic.source_proposition_hub_persistence import (
    SourcePropositionHubPersistenceNode,
)


def add_source_proposition_hub_phase(
    graph: StateGraph,
    proposition: SourcePropositionHubNode,
    persistence: SourcePropositionHubPersistenceNode,
) -> None:
    """Register proposition reranking, judgment, grouping, and persistence."""
    graph.add_node('source_proposition_hub_load', proposition.load_candidates)
    graph.add_node(
        'source_proposition_hub_rerank_worker', proposition.rerank_worker
    )
    graph.add_node(
        'source_proposition_hub_rerank_collect', proposition.collect_rerank
    )
    graph.add_node(
        'source_proposition_hub_judge_worker', proposition.judge_worker
    )
    graph.add_node(
        'source_proposition_hub_judge_collect', proposition.collect_groups
    )
    graph.add_node('source_proposition_hub_prepare', proposition.prepare_hubs)
    graph.add_node('source_proposition_hub_persistence', persistence.run)

    graph.add_conditional_edges(
        'source_proposition_hub_load',
        proposition.dispatch_rerank,
        [
            'source_proposition_hub_rerank_worker',
            'source_proposition_hub_rerank_collect',
        ],
    )
    graph.add_edge(
        'source_proposition_hub_rerank_worker',
        'source_proposition_hub_rerank_collect',
    )
    graph.add_conditional_edges(
        'source_proposition_hub_rerank_collect',
        proposition.dispatch_judge,
        [
            'source_proposition_hub_judge_worker',
            'source_proposition_hub_judge_collect',
        ],
    )
    graph.add_edge(
        'source_proposition_hub_judge_worker',
        'source_proposition_hub_judge_collect',
    )
    graph.add_edge(
        'source_proposition_hub_judge_collect',
        'source_proposition_hub_prepare',
    )
    graph.add_edge(
        'source_proposition_hub_prepare',
        'source_proposition_hub_persistence',
    )
