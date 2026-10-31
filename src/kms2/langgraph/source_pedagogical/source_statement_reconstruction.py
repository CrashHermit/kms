"""LangGraph registration for source statement reconstruction."""

from langgraph.graph import StateGraph

from kms2.node.source_pedagogical.source_statement_reconstruction import (
    SourceStatementReconstructionNode,
)


def add_source_statement_reconstruction_phase(
    graph: StateGraph,
    node: SourceStatementReconstructionNode,
) -> None:
    """Register packet reconstruction and embedding handoff for statements."""
    graph.add_node('source_statement_reconstruction', node.run)
    graph.add_node('source_statement_reconstruction_worker', node.worker)
    graph.add_node('source_statement_reconstruction_collect', node.collect)
    graph.add_edge('instruction_governance', 'source_statement_reconstruction')
    graph.add_conditional_edges(
        'source_statement_reconstruction',
        node.dispatch,
        [
            'source_statement_reconstruction_worker',
            'source_statement_reconstruction_collect',
        ],
    )
    graph.add_edge(
        'source_statement_reconstruction_worker',
        'source_statement_reconstruction_collect',
    )
    graph.add_edge(
        'source_statement_reconstruction_collect',
        'source_statement_embedding',
    )
