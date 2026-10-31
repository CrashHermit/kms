"""LangGraph registration for proposition embeddings."""

from langgraph.graph import StateGraph

from kms2.node.source_semantic.source_proposition_embedding import (
    SourcePropositionEmbeddingNode,
)
from kms2.node.source_semantic.source_proposition_embedding_persistence import (
    SourcePropositionEmbeddingPersistenceNode,
)


def add_source_proposition_embedding_phase(
    graph: StateGraph,
    embedding: SourcePropositionEmbeddingNode,
    persistence: SourcePropositionEmbeddingPersistenceNode,
) -> None:
    """Register proposition embedding inference and persistence."""
    graph.add_node('source_proposition_embedding', embedding.run)
    graph.add_node('source_proposition_embedding_persistence', persistence.run)
    graph.add_edge(
        'source_proposition_persistence', 'source_proposition_embedding'
    )
    graph.add_edge(
        'source_proposition_embedding',
        'source_proposition_embedding_persistence',
    )
