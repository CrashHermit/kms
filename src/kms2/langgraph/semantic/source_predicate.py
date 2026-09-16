"""LangGraph registration for source predicate embedding and persistence."""

from kms2.node.semantic.source_predicate_embedding import (
    SourcePredicateEmbeddingNode,
)
from kms2.node.semantic.source_predicate_persistence import (
    SourcePredicatePersistenceNode,
)
from langgraph.graph import StateGraph


def add_source_predicate_phase(
    graph: StateGraph,
    embedding: SourcePredicateEmbeddingNode,
    persistence: SourcePredicatePersistenceNode,
) -> None:
    """Register source predicate embedding and persistence."""
    graph.add_node('source_predicate_embedding_worker', embedding.worker)
    graph.add_node('source_predicate_embedding_collect', embedding.collect)
    graph.add_node('source_predicate_persistence', persistence.run)
    graph.add_conditional_edges(
        'source_predicate_description_collect',
        embedding.dispatch,
        [
            'source_predicate_embedding_worker',
            'source_predicate_embedding_collect',
        ],
    )
    graph.add_edge(
        'source_predicate_embedding_worker',
        'source_predicate_embedding_collect',
    )
    graph.add_edge(
        'source_predicate_embedding_collect', 'source_predicate_persistence'
    )
