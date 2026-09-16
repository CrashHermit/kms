"""LangGraph registration for source event embedding and persistence."""

from kms2.node.semantic.source_event_embedding import SourceEventEmbeddingNode
from kms2.node.semantic.source_event_persistence import (
    SourceEventPersistenceNode,
)
from langgraph.graph import StateGraph


def add_source_event_phase(
    graph: StateGraph,
    embedding: SourceEventEmbeddingNode,
    persistence: SourceEventPersistenceNode,
) -> None:
    """Register source event embedding and persistence."""
    graph.add_node('source_event_embedding_worker', embedding.worker)
    graph.add_node('source_event_embedding_collect', embedding.collect)
    graph.add_node('source_event_persistence', persistence.run)
    graph.add_conditional_edges(
        'source_event_description_collect',
        embedding.dispatch,
        ['source_event_embedding_worker', 'source_event_embedding_collect'],
    )
    graph.add_edge(
        'source_event_embedding_worker', 'source_event_embedding_collect'
    )
    graph.add_edge('source_event_embedding_collect', 'source_event_persistence')
