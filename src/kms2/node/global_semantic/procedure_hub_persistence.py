"""Persist prepared global procedure hubs."""

from kms2.database.global_semantic.procedure_hub_repository import (
    GlobalProcedureHubRepository,
)
from kms2.langgraph.global_semantic.state import GlobalSemanticState


class GlobalProcedureHubPersistenceNode:
    """Persist global procedure hubs and relationship memberships."""

    def __init__(self, repository: GlobalProcedureHubRepository) -> None:
        self._repository = repository

    async def run(self, state: GlobalSemanticState) -> dict[str, int]:
        """Replace the derived global procedure layer."""
        await self._repository.replace_global_procedure_hubs(
            state.global_procedure_hubs,
            state.global_procedure_hub_memberships,
        )
        return {'global_procedure_hub_count': len(state.global_procedure_hubs)}


__all__ = ['GlobalProcedureHubPersistenceNode']
