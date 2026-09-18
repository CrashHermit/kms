"""Persist prepared source-local event hubs."""

from kms2.database.semantic.source_event_repository import SourceEventRepository
from kms2.langgraph.semantic.state import SemanticState


class SourceEventHubPersistenceNode:
    """Persist source event hubs independently from other hub types."""

    def __init__(
        self,
        repository: SourceEventRepository,
    ) -> None:
        self._repository = repository

    async def run(self, state: SemanticState) -> dict[str, int]:
        await self._repository.replace_source_event_hubs(
            state.source_uuid,
            state.source_event_hubs,
            state.source_event_hub_memberships,
        )
        return {'source_event_hub_count': len(state.source_event_hubs)}
