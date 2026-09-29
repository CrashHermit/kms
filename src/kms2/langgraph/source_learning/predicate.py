"""LangGraph registration for source predicate learning phases."""

from langgraph.graph import StateGraph

from kms2.node.source_learning.predicate import (
    SourcePredicateFlashcardNode,
    SourcePredicateLearningFactNode,
)


def add_source_predicate_learning_phase(
    graph: StateGraph,
    learning_fact: SourcePredicateLearningFactNode,
    flashcard: SourcePredicateFlashcardNode,
) -> None:
    """Register predicate learning-fact and flashcard phases."""
    graph.add_node('source_predicate_learning_fact_load', learning_fact.load)
    graph.add_node(
        'source_predicate_learning_fact_worker', learning_fact.worker
    )
    graph.add_node(
        'source_predicate_learning_fact_collect', learning_fact.collect
    )
    graph.add_node(
        'source_predicate_learning_fact_persistence', learning_fact.persist
    )
    graph.add_node('source_predicate_flashcard_load', flashcard.load)
    graph.add_node('source_predicate_flashcard_worker', flashcard.worker)
    graph.add_node('source_predicate_flashcard_collect', flashcard.collect)
    graph.add_node('source_predicate_flashcard_persistence', flashcard.persist)
    graph.add_conditional_edges(
        'source_predicate_learning_fact_load',
        learning_fact.dispatch,
        [
            'source_predicate_learning_fact_worker',
            'source_predicate_learning_fact_collect',
        ],
    )
    graph.add_edge(
        'source_predicate_learning_fact_worker',
        'source_predicate_learning_fact_collect',
    )
    graph.add_edge(
        'source_predicate_learning_fact_collect',
        'source_predicate_learning_fact_persistence',
    )
    graph.add_edge(
        'source_predicate_learning_fact_persistence',
        'source_predicate_flashcard_load',
    )
    graph.add_conditional_edges(
        'source_predicate_flashcard_load',
        flashcard.dispatch,
        [
            'source_predicate_flashcard_worker',
            'source_predicate_flashcard_collect',
        ],
    )
    graph.add_edge(
        'source_predicate_flashcard_worker',
        'source_predicate_flashcard_collect',
    )
    graph.add_edge(
        'source_predicate_flashcard_collect',
        'source_predicate_flashcard_persistence',
    )
