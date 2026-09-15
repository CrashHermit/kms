"""LangGraph registration for the source embedding phase."""

from kms2.node.source.embedding import EmbeddingNode
from langgraph.graph import StateGraph


def add_embedding_phase(graph: StateGraph, node: EmbeddingNode) -> None:
    """Register embedding dispatch, worker, and collection."""
    graph.add_node('embedding_worker', node.worker)
    graph.add_node('embedding_collect', node.collect)
    graph.add_conditional_edges(
        'instruction_governance',
        node.dispatch,
        ['embedding_worker', 'embedding_collect'],
    )
    graph.add_edge('embedding_worker', 'embedding_collect')
