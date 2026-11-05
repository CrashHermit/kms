"""LangGraph registration for the OCR phase."""

from langgraph.graph import StateGraph

from kms.node.source_processing.ocr import OCRNode


def add_ocr_phase(graph: StateGraph, node: OCRNode) -> None:
    """Register PDF OCR execution."""
    graph.add_node('ocr', node.run)
