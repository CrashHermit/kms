"""LangGraph registration for contextual source-image filtering."""

from langgraph.graph import StateGraph

from kms.node.source_processing.image_filter import ImageFilterNode


def add_image_filter_phase(graph: StateGraph, node: ImageFilterNode) -> None:
    """Register image-filter fan-out, worker, and collection."""
    graph.add_node('image_filter_worker', node.worker)
    graph.add_node('image_filter_collect', node.collect)
    graph.add_conditional_edges(
        'image_seam_odd_collect',
        node.dispatch,
        ['image_filter_worker', 'image_filter_collect'],
    )
    graph.add_edge('image_filter_worker', 'image_filter_collect')
