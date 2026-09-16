"""LangGraph registration for source statement embeddings and persistence."""

from kms2.node.semantic.source_statement_embedding import (
    SourceStatementEmbeddingNode,
)
from kms2.node.semantic.source_statement_persistence import (
    SourceStatementPersistenceNode,
)
from langgraph.graph import StateGraph


def add_source_statement_phase(
    graph: StateGraph,
    embedding: SourceStatementEmbeddingNode,
    persistence: SourceStatementPersistenceNode,
) -> None:
    """Register source statement embedding and persistence."""
    graph.add_node('source_statement_embedding_worker', embedding.worker)
    graph.add_node('source_statement_embedding_collect', embedding.collect)
    graph.add_node('source_statement_persistence', persistence.run)
    graph.add_conditional_edges(
        'source_statement_description_collect',
        embedding.dispatch,
        [
            'source_statement_embedding_worker',
            'source_statement_embedding_collect',
        ],
    )
    graph.add_edge(
        'source_statement_embedding_worker',
        'source_statement_embedding_collect',
    )
    graph.add_edge(
        'source_statement_embedding_collect', 'source_statement_persistence'
    )
