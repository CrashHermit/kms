"""LangGraph registration for staged snapshot finalization."""

from langgraph.graph import END, StateGraph

from kms2.node.source_admission.finalization import (
    SourceAdmissionFinalizationNode,
)


def add_finalization_phase(
    graph: StateGraph, node: SourceAdmissionFinalizationNode
) -> None:
    """Register the terminal durable-file and identity finalization phase."""
    graph.add_node('finalization', node.run)
    graph.add_edge('finalization', END)
