"""LangGraph registration for the content-correction phase."""

from langgraph.graph import StateGraph

from kms2.node.source_processing.content_correction import ContentCorrectionNode


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
