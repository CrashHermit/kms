"""Persist global triplet hubs and memberships."""

from kms2.database.global_semantic.global_triplet_repository import (
    GlobalTripletRepository,
)
from kms2.langgraph.global_semantic.state import GlobalSemanticState


class GlobalTripletHubPersistenceNode:
    """Persist synthesized global triplet hubs."""

    def __init__(self, repository: GlobalTripletRepository) -> None:
        self._repository = repository

    async def run(self, state: GlobalSemanticState) -> dict[str, int]:
        """Replace global triplet hubs and return the persisted count."""
        await self._repository.replace_global_triplet_hubs(
            state.global_triplet_hubs,
            state.global_triplet_hub_memberships,
        )
        return {'global_triplet_hub_count': len(state.global_triplet_hubs)}
