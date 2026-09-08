"""LangGraph registration for the source image seam phase."""

from kms2.node.source.image_seam import ImageSeamNode
from langgraph.graph import StateGraph


def add_image_seam_phase(graph: StateGraph, node: ImageSeamNode) -> None:
    """Register even and odd image seam fan-out and collection passes."""
    graph.add_node('image_seam_even_worker', node.even_worker)
    graph.add_node('image_seam_even_collect', node.even_collect)
    graph.add_node('image_seam_odd_worker', node.odd_worker)
    graph.add_node('image_seam_odd_collect', node.odd_collect)
    graph.add_conditional_edges(
        'text_seam_odd_collect',
        node.dispatch_even,
        ['image_seam_even_worker', 'image_seam_even_collect'],
    )
    graph.add_edge('image_seam_even_worker', 'image_seam_even_collect')
    graph.add_conditional_edges(
        'image_seam_even_collect',
        node.dispatch_odd,
        ['image_seam_odd_worker', 'image_seam_odd_collect'],
    )
    graph.add_edge('image_seam_odd_worker', 'image_seam_odd_collect')
