"""LangGraph registration for the KMS2 persistence phase."""

from kms2.node.source.persistence import SourcePersistenceNode
from langgraph.graph import END, StateGraph


def add_persistence_phase(
    graph: StateGraph,
    node: SourcePersistenceNode,
) -> None:
    """Register terminal persistence after source-block embedding."""
    graph.add_node('persistence', node.run)
    graph.add_edge('embedding_collect', 'persistence')
    graph.add_edge('persistence', END)
