"""LangGraph registration for typed source inputs."""

from langgraph.graph import START, StateGraph

from kms2.node.source_processing.input import SourceInputNode


def add_source_input_phase(graph: StateGraph, node: SourceInputNode) -> None:
    """Route PDFs, prepared OCR, and prepared text to their proper phase."""
    graph.add_node('prepared_ocr', node.load_ocr)
    graph.add_node('prepared_text', node.load_text)
    graph.add_conditional_edges(
        START,
        node.route,
        {'pdf': 'ocr', 'ocr': 'prepared_ocr', 'text': 'prepared_text'},
    )
