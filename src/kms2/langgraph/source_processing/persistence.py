"""LangGraph registration for the KMS2 persistence phase."""

from langgraph.graph import END, StateGraph

from kms2.node.source_processing.persistence import SourcePersistenceNode


def add_persistence_phase(
    graph: StateGraph,
    node: SourcePersistenceNode,
) -> None:
    """Register terminal persistence after source-block embedding."""
    graph.add_node('persistence', node.run)
    graph.add_edge('embedding_collect', 'persistence')
    graph.add_edge('persistence', END)
