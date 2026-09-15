"""LangGraph registration for statement and procedure construction."""

from kms2.node.source.statement_procedure import StatementProcedureNode
from langgraph.graph import StateGraph


def add_statement_procedure_phase(
    graph: StateGraph, node: StatementProcedureNode
) -> None:
    """Register the sequential statement/procedure construction node."""
    graph.add_node('statement_procedure', node.run)
    graph.add_edge('pedagogical_finder', 'statement_procedure')
