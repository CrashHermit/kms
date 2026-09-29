"""Persist prepared global statement hubs."""

from kms2.database.global_semantic.global_statement_hub_repository import (
    GlobalStatementHubRepository,
)
from kms2.langgraph.global_semantic.state import GlobalSemanticState


class GlobalStatementHubPersistenceNode:
    """Persist global statement hubs and relationship memberships."""

    def __init__(self, repository: GlobalStatementHubRepository) -> None:
        self._repository = repository

    async def run(self, state: GlobalSemanticState) -> dict[str, int]:
        """Replace the derived global statement layer."""
        await self._repository.replace_global_statement_hubs(
            state.global_statement_hubs,
            state.global_statement_hub_memberships,
        )
        return {'global_statement_hub_count': len(state.global_statement_hubs)}
