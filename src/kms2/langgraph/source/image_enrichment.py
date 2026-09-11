"""LangGraph registration for source image enrichment."""

from kms2.node.source.image_enrichment import ImageEnrichmentNode
from langgraph.graph import StateGraph


def add_image_enrichment_phase(
    graph: StateGraph, node: ImageEnrichmentNode
) -> None:
    """Register image enrichment fan-out, worker, and collection."""
    graph.add_node('image_enrichment_worker', node.worker)
    graph.add_node('image_enrichment_collect', node.collect)
    graph.add_conditional_edges(
        'image_seam_odd_collect',
        node.dispatch,
        ['image_enrichment_worker', 'image_enrichment_collect'],
    )
    graph.add_edge('image_enrichment_worker', 'image_enrichment_collect')
