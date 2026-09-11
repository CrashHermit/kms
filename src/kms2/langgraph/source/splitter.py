"""LangGraph registration for the source splitter phase."""

from kms2.node.source.splitter import SplitterNode
from langgraph.graph import StateGraph


def add_splitter_phase(graph: StateGraph, node: SplitterNode) -> None:
    """Register splitter fan-out, worker, and collection."""
    graph.add_node('splitter_worker', node.worker)
    graph.add_node('splitter_collect', node.collect)
    graph.add_conditional_edges(
        'image_enrichment_collect',
        node.dispatch,
        ['splitter_worker', 'splitter_collect'],
    )
    graph.add_edge('splitter_worker', 'splitter_collect')
