"""LangGraph registration for atomic source-learning generation."""

from langgraph.graph import StateGraph

from kms2.node.source_learning.atomic import SourceAtomicFlashcardNode
from kms2.node.source_learning.atomic_persistence import (
    SourceAtomicFlashcardPersistenceNode,
)


def add_source_atomic_flashcard_phase(
    graph: StateGraph,
    node: SourceAtomicFlashcardNode,
    persistence: SourceAtomicFlashcardPersistenceNode,
) -> None:
    """Register load, fanout, collection, and persistence for atomic cards."""
    graph.add_node('source_atomic_flashcard_load', node.load)
    graph.add_node('source_atomic_flashcard_worker', node.worker)
    graph.add_node('source_atomic_flashcard_collect', node.collect)
    graph.add_node('source_atomic_flashcard_persistence', persistence.run)
    graph.add_conditional_edges(
        'source_atomic_flashcard_load',
        node.dispatch,
        [
            'source_atomic_flashcard_worker',
            'source_atomic_flashcard_collect',
        ],
    )
    graph.add_edge(
        'source_atomic_flashcard_worker',
        'source_atomic_flashcard_collect',
    )
    graph.add_edge(
        'source_atomic_flashcard_collect',
        'source_atomic_flashcard_persistence',
    )
