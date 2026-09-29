"""LangGraph registration for instruction governance."""

from langgraph.graph import StateGraph

from kms2.node.source_processing.instruction_governance import (
    InstructionGovernanceNode,
)


def add_instruction_governance_phase(
    graph: StateGraph,
    node: InstructionGovernanceNode,
) -> None:
    """Register the sequential instruction governance node."""
    graph.add_node('instruction_governance', node.run)
    graph.add_edge('statement_procedure', 'instruction_governance')
