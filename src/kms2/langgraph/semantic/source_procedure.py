"""LangGraph registration for source procedure embeddings and persistence."""

from kms2.node.semantic.source_procedure_embedding import (
    SourceProcedureEmbeddingNode,
)
from kms2.node.semantic.source_procedure_persistence import (
    SourceProcedurePersistenceNode,
)
from langgraph.graph import StateGraph


def add_source_procedure_phase(
    graph: StateGraph,
    embedding: SourceProcedureEmbeddingNode,
    persistence: SourceProcedurePersistenceNode,
) -> None:
    """Register source procedure embedding and persistence."""
    graph.add_node('source_procedure_embedding_worker', embedding.worker)
    graph.add_node('source_procedure_embedding_collect', embedding.collect)
    graph.add_node('source_procedure_persistence', persistence.run)
    graph.add_conditional_edges(
        'source_procedure_description_collect',
        embedding.dispatch,
        [
            'source_procedure_embedding_worker',
            'source_procedure_embedding_collect',
        ],
    )
    graph.add_edge(
        'source_procedure_embedding_worker',
        'source_procedure_embedding_collect',
    )
    graph.add_edge(
        'source_procedure_embedding_collect', 'source_procedure_persistence'
    )
