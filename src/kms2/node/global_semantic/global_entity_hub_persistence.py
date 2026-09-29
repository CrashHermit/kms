"""Persist prepared global entity hubs."""

from kms2.database.global_semantic.global_entity_hub_repository import (
    GlobalEntityHubRepository,
)
from kms2.langgraph.global_semantic.state import GlobalSemanticState


class GlobalEntityHubPersistenceNode:
    """Persist global entity hubs and relationship memberships."""

    def __init__(self, repository: GlobalEntityHubRepository) -> None:
        self._repository = repository

    async def run(self, state: GlobalSemanticState) -> dict[str, int]:
        """Replace the derived global entity layer."""
        await self._repository.replace_global_entity_hubs(
            state.global_entity_hubs,
            state.global_entity_hub_memberships,
        )
        return {'global_entity_hub_count': len(state.global_entity_hubs)}
