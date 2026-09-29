"""LangGraph registration for source triplet learning phases."""

from langgraph.graph import StateGraph

from kms2.node.source_learning.triplet import (
    SourceTripletFlashcardNode,
    SourceTripletLearningFactNode,
)


def add_source_triplet_learning_phase(
    graph: StateGraph,
    learning_fact: SourceTripletLearningFactNode,
    flashcard: SourceTripletFlashcardNode,
) -> None:
    """Register triplet learning-fact and flashcard phases."""
    graph.add_node('source_triplet_learning_fact_load', learning_fact.load)
    graph.add_node('source_triplet_learning_fact_worker', learning_fact.worker)
    graph.add_node(
        'source_triplet_learning_fact_collect', learning_fact.collect
    )
    graph.add_node(
        'source_triplet_learning_fact_persistence', learning_fact.persist
    )
    graph.add_node('source_triplet_flashcard_load', flashcard.load)
    graph.add_node('source_triplet_flashcard_worker', flashcard.worker)
    graph.add_node('source_triplet_flashcard_collect', flashcard.collect)
    graph.add_node('source_triplet_flashcard_persistence', flashcard.persist)
    graph.add_conditional_edges(
        'source_triplet_learning_fact_load',
        learning_fact.dispatch,
        [
            'source_triplet_learning_fact_worker',
            'source_triplet_learning_fact_collect',
        ],
    )
    graph.add_edge(
        'source_triplet_learning_fact_worker',
        'source_triplet_learning_fact_collect',
    )
    graph.add_edge(
        'source_triplet_learning_fact_collect',
        'source_triplet_learning_fact_persistence',
    )
    graph.add_edge(
        'source_triplet_learning_fact_persistence',
        'source_triplet_flashcard_load',
    )
    graph.add_conditional_edges(
        'source_triplet_flashcard_load',
        flashcard.dispatch,
        ['source_triplet_flashcard_worker', 'source_triplet_flashcard_collect'],
    )
    graph.add_edge(
        'source_triplet_flashcard_worker', 'source_triplet_flashcard_collect'
    )
    graph.add_edge(
        'source_triplet_flashcard_collect',
        'source_triplet_flashcard_persistence',
    )
