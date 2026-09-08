"""LangGraph registration for the OCR phase."""

from kms2.node.source.ocr import OCRNode
from langgraph.graph import START, StateGraph


def add_ocr_phase(graph: StateGraph, node: OCRNode) -> None:
    """Register OCR execution and its entry edge."""
    graph.add_node('ocr', node.run)
    graph.add_edge(START, 'ocr')
