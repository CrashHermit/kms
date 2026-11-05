"""LangGraph registration for the source embedding phase."""

from langgraph.graph import StateGraph

from kms.node.source_processing.embedding import EmbeddingNode


def add_embedding_phase(graph: StateGraph, node: EmbeddingNode) -> None:
    """Register direct source-block embedding before persistence."""
    graph.add_node('embedding', node.run)
    graph.add_edge('image_description_collect', 'embedding')
