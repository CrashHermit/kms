"""Persist prepared source-local procedure hubs."""

from kms2.database.source_semantic.source_procedure_repository import (
    SourceProcedureRepository,
)
from kms2.langgraph.source_semantic.state import SourceSemanticState


class SourceProcedureHubPersistenceNode:
    """Persist source procedure hubs independently from other hub types."""

    def __init__(
        self,
        repository: SourceProcedureRepository,
    ) -> None:
        self._repository = repository

    async def run(self, state: SourceSemanticState) -> dict[str, int]:
        await self._repository.replace_source_procedure_hubs(
            state.source_uuid,
            state.source_procedure_hubs,
            state.source_procedure_hub_memberships,
        )
        return {'source_procedure_hub_count': len(state.source_procedure_hubs)}
