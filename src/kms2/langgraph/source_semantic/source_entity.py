"""LangGraph registration for source entity embedding and persistence."""

from langgraph.graph import StateGraph

from kms2.node.source_semantic.source_entity_embedding import (
    SourceEntityEmbeddingNode,
)
from kms2.node.source_semantic.source_entity_persistence import (
    SourceEntityPersistenceNode,
)


def add_source_entity_phase(
    graph: StateGraph,
    embedding: SourceEntityEmbeddingNode,
    persistence: SourceEntityPersistenceNode,
) -> None:
    """Register source entity embedding and persistence."""
    graph.add_node('source_entity_embedding', embedding.run)
    graph.add_node('source_entity_persistence', persistence.run)
    graph.add_edge(
        'source_entity_description_collect', 'source_entity_embedding'
    )
    graph.add_edge('source_entity_embedding', 'source_entity_persistence')
