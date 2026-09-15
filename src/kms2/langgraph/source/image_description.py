"""LangGraph registration for source image descriptions."""

from kms2.node.source.image_description import ImageDescriptionNode
from langgraph.graph import StateGraph


def add_image_description_phase(
    graph: StateGraph, node: ImageDescriptionNode
) -> None:
    """Register image description fan-out, worker, and collection."""
    graph.add_node('image_description_worker', node.worker)
    graph.add_node('image_description_collect', node.collect)
    graph.add_conditional_edges(
        'image_seam_odd_collect',
        node.dispatch,
        ['image_description_worker', 'image_description_collect'],
    )
    graph.add_edge('image_description_worker', 'image_description_collect')
