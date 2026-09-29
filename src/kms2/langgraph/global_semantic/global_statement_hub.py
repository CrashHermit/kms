"""LangGraph registration for global statement hub discovery."""

from langgraph.graph import StateGraph

from kms2.node.global_semantic.global_statement_hub import (
    GlobalStatementHubNode,
)
from kms2.node.global_semantic.global_statement_hub_persistence import (
    GlobalStatementHubPersistenceNode,
)


def add_global_statement_hub_phase(
    graph: StateGraph,
    statement: GlobalStatementHubNode,
    persistence: GlobalStatementHubPersistenceNode,
) -> None:
    """Register global statement hub discovery and persistence."""
    graph.add_node('global_statement_hub_load', statement.load_candidates)
    graph.add_node(
        'global_statement_hub_rerank_worker', statement.rerank_worker
    )
    graph.add_node(
        'global_statement_hub_rerank_collect', statement.collect_rerank
    )
    graph.add_node('global_statement_hub_judge_worker', statement.judge_worker)
    graph.add_node(
        'global_statement_hub_judge_collect', statement.collect_judge
    )
    graph.add_node(
        'global_statement_hub_communities', statement.detect_communities
    )
    graph.add_node(
        'global_statement_hub_synthesis_worker', statement.synthesis_worker
    )
    graph.add_node(
        'global_statement_hub_synthesis_collect', statement.collect_synthesis
    )
    graph.add_node('global_statement_hub_embedding', statement.embed)
    graph.add_node('global_statement_hub_persistence', persistence.run)

    graph.add_conditional_edges(
        'global_statement_hub_load',
        statement.dispatch_rerank,
        [
            'global_statement_hub_rerank_worker',
            'global_statement_hub_rerank_collect',
        ],
    )
    graph.add_edge(
        'global_statement_hub_rerank_worker',
        'global_statement_hub_rerank_collect',
    )
    graph.add_conditional_edges(
        'global_statement_hub_rerank_collect',
        statement.dispatch_judge,
        [
            'global_statement_hub_judge_worker',
            'global_statement_hub_judge_collect',
        ],
    )
    graph.add_edge(
        'global_statement_hub_judge_worker',
        'global_statement_hub_judge_collect',
    )
    graph.add_edge(
        'global_statement_hub_judge_collect',
        'global_statement_hub_communities',
    )
    graph.add_conditional_edges(
        'global_statement_hub_communities',
        statement.dispatch_synthesis,
        [
            'global_statement_hub_synthesis_worker',
            'global_statement_hub_synthesis_collect',
        ],
    )
    graph.add_edge(
        'global_statement_hub_synthesis_worker',
        'global_statement_hub_synthesis_collect',
    )
    graph.add_edge(
        'global_statement_hub_synthesis_collect',
        'global_statement_hub_embedding',
    )
    graph.add_edge(
        'global_statement_hub_embedding', 'global_statement_hub_persistence'
    )
