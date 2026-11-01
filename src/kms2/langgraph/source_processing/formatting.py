"""LangGraph registration for the source-content formatting phase."""

from langgraph.graph import StateGraph

from kms2.node.source_processing.formatting import FormattingNode


def add_formatter_phase(graph: StateGraph, node: FormattingNode) -> None:
    """Register formatter fan-out, worker, and collector."""
    graph.add_node('formatter_worker', node.worker)
    graph.add_node('formatter_collect', node.collect)
    for source in ('content_correction_collect', 'prepared_text'):
        graph.add_conditional_edges(
            source,
            node.dispatch,
            ['formatter_worker', 'formatter_collect'],
        )
    graph.add_edge('formatter_worker', 'formatter_collect')
