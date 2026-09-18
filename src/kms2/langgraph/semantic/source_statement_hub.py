"""LangGraph registration for source statement hub discovery."""

from kms2.node.semantic.source_statement_hub import SourceStatementHubNode
from kms2.node.semantic.source_statement_hub_persistence import (
    SourceStatementHubPersistenceNode,
)
from langgraph.graph import StateGraph


def add_source_statement_hub_phase(
    graph: StateGraph,
    statement: SourceStatementHubNode,
    persistence: SourceStatementHubPersistenceNode,
) -> None:
    """Register source statement hub discovery, synthesis, and persistence."""
    graph.add_node('source_statement_hub_load', statement.load_candidates)
    graph.add_node(
        'source_statement_hub_rerank_worker', statement.rerank_worker
    )
    graph.add_node(
        'source_statement_hub_rerank_collect', statement.collect_rerank
    )
    graph.add_node('source_statement_hub_judge_worker', statement.judge_worker)
    graph.add_node(
        'source_statement_hub_judge_collect', statement.collect_judge
    )
    graph.add_node(
        'source_statement_hub_communities', statement.detect_communities
    )
    graph.add_node(
        'source_statement_hub_synthesis_worker', statement.synthesis_worker
    )
    graph.add_node(
        'source_statement_hub_synthesis_collect', statement.collect_synthesis
    )
    graph.add_node('source_statement_hub_embedding', statement.embed)
    graph.add_node('source_statement_hub_persistence', persistence.run)

    graph.add_conditional_edges(
        'source_statement_hub_load',
        statement.dispatch_rerank,
        [
            'source_statement_hub_rerank_worker',
            'source_statement_hub_rerank_collect',
        ],
    )
    graph.add_edge(
        'source_statement_hub_rerank_worker',
        'source_statement_hub_rerank_collect',
    )
    graph.add_conditional_edges(
        'source_statement_hub_rerank_collect',
        statement.dispatch_judge,
        [
            'source_statement_hub_judge_worker',
            'source_statement_hub_judge_collect',
        ],
    )
    graph.add_edge(
        'source_statement_hub_judge_worker',
        'source_statement_hub_judge_collect',
    )
    graph.add_edge(
        'source_statement_hub_judge_collect',
        'source_statement_hub_communities',
    )
    graph.add_conditional_edges(
        'source_statement_hub_communities',
        statement.dispatch_synthesis,
        [
            'source_statement_hub_synthesis_worker',
            'source_statement_hub_synthesis_collect',
        ],
    )
    graph.add_edge(
        'source_statement_hub_synthesis_worker',
        'source_statement_hub_synthesis_collect',
    )
    graph.add_edge(
        'source_statement_hub_synthesis_collect',
        'source_statement_hub_embedding',
    )
    graph.add_edge(
        'source_statement_hub_embedding', 'source_statement_hub_persistence'
    )
