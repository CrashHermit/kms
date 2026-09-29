"""Persist prepared global predicate hubs."""

from kms2.database.global_semantic.global_predicate_hub_repository import (
    GlobalPredicateHubRepository,
)
from kms2.langgraph.global_semantic.state import GlobalSemanticState


class GlobalPredicateHubPersistenceNode:
    """Persist global predicate hubs and relationship memberships."""

    def __init__(self, repository: GlobalPredicateHubRepository) -> None:
        self._repository = repository

    async def run(self, state: GlobalSemanticState) -> dict[str, int]:
        """Replace the derived global predicate layer."""
        await self._repository.replace_global_predicate_hubs(
            state.global_predicate_hubs,
            state.global_predicate_hub_memberships,
        )
        return {'global_predicate_hub_count': len(state.global_predicate_hubs)}
