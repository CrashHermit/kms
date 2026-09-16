"""LangGraph registration for source entity descriptions."""

from kms2.node.semantic.source_entity_description import (
    SourceEntityDescriptionNode,
)
from kms2.node.semantic.source_entity_description_load import (
    SourceEntityDescriptionLoadNode,
)
from langgraph.graph import StateGraph


def add_source_entity_description_phase(
    graph: StateGraph,
    description_load: SourceEntityDescriptionLoadNode,
    description: SourceEntityDescriptionNode,
) -> None:
    """Register source entity description loading and generation."""
    graph.add_node('source_entity_description_load', description_load.run)
    graph.add_node('source_entity_description_worker', description.worker)
    graph.add_node('source_entity_description_collect', description.collect)
    graph.add_edge('triplet_persistence', 'source_entity_description_load')
    graph.add_conditional_edges(
        'source_entity_description_load',
        description.dispatch,
        [
            'source_entity_description_worker',
            'source_entity_description_collect',
        ],
    )
    graph.add_edge(
        'source_entity_description_worker', 'source_entity_description_collect'
    )
