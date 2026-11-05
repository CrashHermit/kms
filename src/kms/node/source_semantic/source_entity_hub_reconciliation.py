"""Run iterative source-local entity hub reconciliation."""

from kms.langgraph.source_semantic.entity_hub.graph import SourceEntityHubGraph
from kms.langgraph.source_semantic.state import SourceSemanticState
from kms.node.source_semantic.source_entity_hub import SourceEntityHubNode


class SourceEntityHubReconciliationNode:
    """Initialize entity members, reconcile them, and project hub updates."""

    def __init__(
        self, entity: SourceEntityHubNode, graph: SourceEntityHubGraph
    ) -> None:
        self._entity = entity
        self._graph = graph

    async def run(self, state: SourceSemanticState) -> dict[str, object]:
        initial = await self._entity.initialize(state)
        initial_state = SourceSemanticState(
            source_uuid=state.source_uuid,
            **initial,
        )
        result = await self._graph.run(initial_state)
        return {
            'source_entity_hubs': result.source_entity_hubs,
            'source_entity_hub_memberships': result.source_entity_hub_memberships,
            'source_entity_hub_count': len(result.source_entity_hubs),
        }
