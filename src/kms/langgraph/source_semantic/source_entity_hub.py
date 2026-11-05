"""Register the source entity hub reconciliation and persistence phase."""

from langgraph.graph import StateGraph

from kms.node.source_semantic.source_entity_hub_persistence import (
    SourceEntityHubPersistenceNode,
)
from kms.node.source_semantic.source_entity_hub_reconciliation import (
    SourceEntityHubReconciliationNode,
)


def add_source_entity_hub_phase(
    graph: StateGraph,
    reconciliation: SourceEntityHubReconciliationNode,
    persistence: SourceEntityHubPersistenceNode,
) -> None:
    """Register source-local reconciliation followed by hub persistence."""
    graph.add_node('source_entity_hub_load', reconciliation.run)
    graph.add_node('source_entity_hub_persistence', persistence.run)
    graph.add_edge('source_entity_hub_load', 'source_entity_hub_persistence')
