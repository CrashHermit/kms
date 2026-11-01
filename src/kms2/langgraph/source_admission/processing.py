"""LangGraph registration for ordinary source processing."""

from langgraph.graph import StateGraph

from kms2.node.source_admission.processing import SourceAdmissionProcessingNode


def add_processing_phase(
    graph: StateGraph, node: SourceAdmissionProcessingNode
) -> None:
    """Register ordinary source processing before identity finalization."""
    graph.add_node('processing', node.run)
    graph.add_edge('processing', 'finalization')
