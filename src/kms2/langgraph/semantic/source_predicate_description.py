"""LangGraph registration for source predicate descriptions."""

from kms2.node.semantic.source_predicate_description import (
    SourcePredicateDescriptionNode,
)
from kms2.node.semantic.source_predicate_description_load import (
    SourcePredicateDescriptionLoadNode,
)
from langgraph.graph import StateGraph


def add_source_predicate_description_phase(
    graph: StateGraph,
    description_load: SourcePredicateDescriptionLoadNode,
    description: SourcePredicateDescriptionNode,
) -> None:
    """Register source predicate description loading and generation."""
    graph.add_node('source_predicate_description_load', description_load.run)
    graph.add_node('source_predicate_description_worker', description.worker)
    graph.add_node('source_predicate_description_collect', description.collect)
    graph.add_edge('triplet_persistence', 'source_predicate_description_load')
    graph.add_conditional_edges(
        'source_predicate_description_load',
        description.dispatch,
        [
            'source_predicate_description_worker',
            'source_predicate_description_collect',
        ],
    )
    graph.add_edge(
        'source_predicate_description_worker',
        'source_predicate_description_collect',
    )
