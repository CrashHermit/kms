"""LangGraph registration for instruction governance."""

from kms2.node.source.instruction_governance import InstructionGovernanceNode
from langgraph.graph import StateGraph


def add_instruction_governance_phase(
    graph: StateGraph,
    node: InstructionGovernanceNode,
) -> None:
    """Register the sequential instruction governance node."""
    graph.add_node('instruction_governance', node.run)
    graph.add_edge('statement_procedure', 'instruction_governance')
