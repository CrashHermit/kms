"""Persist prepared global event hubs."""

from kms2.database.global_semantic.event_hub_repository import (
    GlobalEventHubRepository,
)
from kms2.langgraph.global_semantic.state import GlobalSemanticState


class GlobalEventHubPersistenceNode:
    """Persist global event hubs and relationship memberships."""

    def __init__(self, repository: GlobalEventHubRepository) -> None:
        self._repository = repository

    async def run(self, state: GlobalSemanticState) -> dict[str, int]:
        """Replace the derived global event layer."""
        await self._repository.replace_global_event_hubs(
            state.global_event_hubs,
            state.global_event_hub_memberships,
        )
        return {'global_event_hub_count': len(state.global_event_hubs)}


__all__ = ['GlobalEventHubPersistenceNode']
