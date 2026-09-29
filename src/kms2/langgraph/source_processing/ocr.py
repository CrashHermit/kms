"""LangGraph registration for the OCR phase."""

from langgraph.graph import START, StateGraph

from kms2.node.source_processing.ocr import OCRNode


def add_ocr_phase(graph: StateGraph, node: OCRNode) -> None:
    """Register OCR execution and its entry edge."""
    graph.add_node('ocr', node.run)
    graph.add_edge(START, 'ocr')
