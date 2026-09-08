"""LangGraph registration for the content-correction phase."""

from kms2.node.source.content_correction import ContentCorrectionNode
from langgraph.graph import StateGraph


def add_content_correction_phase(
    graph: StateGraph,
    node: ContentCorrectionNode,
) -> None:
    """Register correction fan-out, worker, and collector."""
    graph.add_node('content_correction_worker', node.worker)
    graph.add_node('content_correction_collect', node.collect)
    graph.add_conditional_edges(
        'ocr',
        node.dispatch,
        ['content_correction_worker', 'content_correction_collect'],
    )
    graph.add_edge(
        'content_correction_worker',
        'content_correction_collect',
    )
