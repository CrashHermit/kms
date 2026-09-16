"""LangGraph registration for source entity embedding and persistence."""

from kms2.node.semantic.source_entity_embedding import SourceEntityEmbeddingNode
from kms2.node.semantic.source_entity_persistence import (
    SourceEntityPersistenceNode,
)
from langgraph.graph import StateGraph


def add_source_entity_phase(
    graph: StateGraph,
    embedding: SourceEntityEmbeddingNode,
    persistence: SourceEntityPersistenceNode,
) -> None:
    """Register source entity embedding and persistence."""
    graph.add_node('source_entity_embedding_worker', embedding.worker)
    graph.add_node('source_entity_embedding_collect', embedding.collect)
    graph.add_node('source_entity_persistence', persistence.run)
    graph.add_conditional_edges(
        'source_entity_description_collect',
        embedding.dispatch,
        ['source_entity_embedding_worker', 'source_entity_embedding_collect'],
    )
    graph.add_edge(
        'source_entity_embedding_worker', 'source_entity_embedding_collect'
    )
    graph.add_edge(
        'source_entity_embedding_collect', 'source_entity_persistence'
    )
