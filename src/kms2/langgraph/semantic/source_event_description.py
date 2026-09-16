"""LangGraph registration for source event descriptions."""

from kms2.node.semantic.source_event_description import (
    SourceEventDescriptionNode,
)
from kms2.node.semantic.source_event_description_load import (
    SourceEventDescriptionLoadNode,
)
from langgraph.graph import StateGraph


def add_source_event_description_phase(
    graph: StateGraph,
    description_load: SourceEventDescriptionLoadNode,
    description: SourceEventDescriptionNode,
) -> None:
    """Register source event description loading and generation."""
    graph.add_node('source_event_description_load', description_load.run)
    graph.add_node('source_event_description_worker', description.worker)
    graph.add_node('source_event_description_collect', description.collect)
    graph.add_edge('triplet_persistence', 'source_event_description_load')
    graph.add_conditional_edges(
        'source_event_description_load',
        description.dispatch,
        ['source_event_description_worker', 'source_event_description_collect'],
    )
    graph.add_edge(
        'source_event_description_worker', 'source_event_description_collect'
    )
