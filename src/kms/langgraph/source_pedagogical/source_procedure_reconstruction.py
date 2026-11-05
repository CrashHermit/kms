"""LangGraph registration for source procedure reconstruction."""

from langgraph.graph import StateGraph

from kms.node.source_pedagogical.source_procedure_reconstruction import (
    SourceProcedureReconstructionNode,
)


def add_source_procedure_reconstruction_phase(
    graph: StateGraph,
    node: SourceProcedureReconstructionNode,
) -> None:
    """Register packet reconstruction and embedding handoff for procedures."""
    graph.add_node('source_procedure_reconstruction', node.run)
    graph.add_node('source_procedure_reconstruction_worker', node.worker)
    graph.add_node('source_procedure_reconstruction_collect', node.collect)
    graph.add_edge('instruction_governance', 'source_procedure_reconstruction')
    graph.add_conditional_edges(
        'source_procedure_reconstruction',
        node.dispatch,
        [
            'source_procedure_reconstruction_worker',
            'source_procedure_reconstruction_collect',
        ],
    )
    graph.add_edge(
        'source_procedure_reconstruction_worker',
        'source_procedure_reconstruction_collect',
    )
    graph.add_edge(
        'source_procedure_reconstruction_collect',
        'source_procedure_embedding',
    )
