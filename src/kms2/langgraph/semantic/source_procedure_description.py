"""LangGraph registration for source procedure descriptions."""

from kms2.node.semantic.source_procedure_description import (
    SourceProcedureDescriptionNode,
)
from kms2.node.semantic.source_procedure_description_load import (
    SourceProcedureDescriptionLoadNode,
)
from langgraph.graph import StateGraph


def add_source_procedure_description_phase(
    graph: StateGraph,
    description_load: SourceProcedureDescriptionLoadNode,
    description: SourceProcedureDescriptionNode,
) -> None:
    """Register source procedure description loading and generation."""
    graph.add_node('source_procedure_description_load', description_load.run)
    graph.add_node('source_procedure_description_worker', description.worker)
    graph.add_node('source_procedure_description_collect', description.collect)
    graph.add_edge(
        'source_statement_persistence', 'source_procedure_description_load'
    )
    graph.add_conditional_edges(
        'source_procedure_description_load',
        description.dispatch,
        [
            'source_procedure_description_worker',
            'source_procedure_description_collect',
        ],
    )
    graph.add_edge(
        'source_procedure_description_worker',
        'source_procedure_description_collect',
    )
