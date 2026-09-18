"""LangGraph registration for source statement descriptions."""

from kms2.node.semantic.source_statement_description import (
    SourceStatementDescriptionNode,
)
from kms2.node.semantic.source_statement_description_load import (
    SourceStatementDescriptionLoadNode,
)
from langgraph.graph import StateGraph


def add_source_statement_description_phase(
    graph: StateGraph,
    description_load: SourceStatementDescriptionLoadNode,
    description: SourceStatementDescriptionNode,
) -> None:
    """Register source statement description loading and generation."""
    graph.add_node('source_statement_description_load', description_load.run)
    graph.add_node('source_statement_description_worker', description.worker)
    graph.add_node('source_statement_description_collect', description.collect)
    graph.add_edge(
        'source_predicate_persistence', 'source_statement_description_load'
    )
    graph.add_conditional_edges(
        'source_statement_description_load',
        description.dispatch,
        [
            'source_statement_description_worker',
            'source_statement_description_collect',
        ],
    )
    graph.add_edge(
        'source_statement_description_worker',
        'source_statement_description_collect',
    )
