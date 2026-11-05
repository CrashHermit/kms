"""LangGraph registration for entity extraction and persistence."""

from langgraph.graph import StateGraph

from kms.node.source_semantic.source_entity_extraction import (
    SourceEntityExtractionNode,
)
from kms.node.source_semantic.source_entity_extraction_persistence import (
    SourceEntityExtractionPersistenceNode,
)


def add_source_entity_extraction_phase(
    graph: StateGraph,
    extraction: SourceEntityExtractionNode,
    persistence: SourceEntityExtractionPersistenceNode,
) -> None:
    """Register source entity extraction workers and raw persistence."""
    graph.add_node(
        'source_entity_extraction_worker',
        extraction.worker,
    )
    graph.add_node(
        'source_entity_extraction_collect',
        extraction.collect,
    )
    graph.add_node(
        'source_entity_extraction_persistence',
        persistence.run,
    )
    graph.add_conditional_edges(
        'source_semantic_load',
        extraction.dispatch_entities,
        [
            'source_entity_extraction_worker',
            'source_entity_extraction_collect',
        ],
    )
    graph.add_edge(
        'source_entity_extraction_worker',
        'source_entity_extraction_collect',
    )
    graph.add_edge(
        'source_entity_extraction_collect',
        'source_entity_extraction_persistence',
    )
