"""Persist prepared source-local entity hubs."""

from kms2.database.source_semantic.source_entity_repository import (
    SourceEntityRepository,
)
from kms2.langgraph.source_semantic.state import SourceSemanticState


class SourceEntityHubPersistenceNode:
    """Persist source entity hubs independently from other hub types."""

    def __init__(
        self,
        repository: SourceEntityRepository,
    ) -> None:
        self._repository = repository

    async def run(self, state: SourceSemanticState) -> dict[str, int]:
        await self._repository.replace_source_entity_hubs(
            state.source_uuid,
            state.source_entity_hubs,
            state.source_entity_hub_memberships,
        )
        return {'source_entity_hub_count': len(state.source_entity_hubs)}
