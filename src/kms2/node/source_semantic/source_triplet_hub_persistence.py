"""Persist prepared source-local triplet hubs."""

from kms2.database.source_semantic.source_triplet_repository import (
    SourceTripletRepository,
)
from kms2.langgraph.source_semantic.state import SourceSemanticState


class SourceTripletHubPersistenceNode:
    """Persist source triplet hubs independently from other hub types."""

    def __init__(
        self,
        repository: SourceTripletRepository,
    ) -> None:
        self._repository = repository

    async def run(self, state: SourceSemanticState) -> dict[str, int]:
        await self._repository.replace_source_triplet_hubs(
            state.source_uuid,
            state.source_triplet_hubs,
            state.source_triplet_hub_memberships,
        )
        return {'source_triplet_hub_count': len(state.source_triplet_hubs)}
