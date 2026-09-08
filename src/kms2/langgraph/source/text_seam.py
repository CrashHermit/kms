"""LangGraph registration for the source text seam phase."""

from kms2.node.source.text_seam import TextSeamNode
from langgraph.graph import StateGraph


def add_text_seam_phase(graph: StateGraph, node: TextSeamNode) -> None:
    """Register even and odd text seam fan-out and collection passes."""
    graph.add_node('text_seam_even_worker', node.even_worker)
    graph.add_node('text_seam_even_collect', node.even_collect)
    graph.add_node('text_seam_odd_worker', node.odd_worker)
    graph.add_node('text_seam_odd_collect', node.odd_collect)
    graph.add_conditional_edges(
        'formatter_collect',
        node.dispatch_even,
        ['text_seam_even_worker', 'text_seam_even_collect'],
    )
    graph.add_edge('text_seam_even_worker', 'text_seam_even_collect')
    graph.add_conditional_edges(
        'text_seam_even_collect',
        node.dispatch_odd,
        ['text_seam_odd_worker', 'text_seam_odd_collect'],
    )
    graph.add_edge('text_seam_odd_worker', 'text_seam_odd_collect')
