"""LangGraph registration for snapshot preparation and lookup."""

from langgraph.graph import END, StateGraph

from kms2.node.source_admission.preparation import (
    SourceAdmissionPreparationNode,
)


def add_preparation_phase(
    graph: StateGraph, node: SourceAdmissionPreparationNode
) -> None:
    """Register the preparation node and existing-snapshot terminal path."""
    graph.add_node('preparation', node.run)
    graph.add_conditional_edges(
        'preparation',
        node.route,
        {'existing': END, 'process': 'processing'},
    )
