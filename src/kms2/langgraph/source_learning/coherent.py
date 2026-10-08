"""LangGraph registration for coherent source-learning generation."""

from langgraph.graph import StateGraph

from kms2.node.source_learning.coherent import SourceCoherentFlashcardNode
from kms2.node.source_learning.coherent_persistence import (
    SourceCoherentFlashcardPersistenceNode,
)


def add_source_coherent_flashcard_phase(
    graph: StateGraph,
    node: SourceCoherentFlashcardNode,
    persistence: SourceCoherentFlashcardPersistenceNode,
) -> None:
    """Register preparation, fanout, collection, and persistence."""
    graph.add_node('source_coherent_flashcard_prepare', node.prepare)
    graph.add_node('source_coherent_flashcard_worker', node.worker)
    graph.add_node('source_coherent_flashcard_collect', node.collect)
    graph.add_node('source_coherent_flashcard_persistence', persistence.run)
    graph.add_conditional_edges(
        'source_coherent_flashcard_prepare',
        node.dispatch,
        [
            'source_coherent_flashcard_worker',
            'source_coherent_flashcard_collect',
        ],
    )
    graph.add_edge(
        'source_coherent_flashcard_worker',
        'source_coherent_flashcard_collect',
    )
    graph.add_edge(
        'source_coherent_flashcard_collect',
        'source_coherent_flashcard_persistence',
    )
