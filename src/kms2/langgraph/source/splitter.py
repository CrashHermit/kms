"""LangGraph registration for the source splitter phase."""

from kms2.node.source.splitter import SplitterNode
from langgraph.graph import END, StateGraph


def add_splitter_phase(graph: StateGraph, node: SplitterNode) -> None:
    """Register routed splitting after image seam collection."""
    graph.add_node('splitter', node.run)
    graph.add_edge('image_seam_odd_collect', 'splitter')
    graph.add_edge('splitter', END)
