"""Persist prepared source-local statement hubs."""

from kms2.database.source_semantic.source_statement_repository import (
    SourceStatementRepository,
)
from kms2.langgraph.source_semantic.state import SourceSemanticState


class SourceStatementHubPersistenceNode:
    """Persist source statement hubs independently from other hub types."""

    def __init__(
        self,
        repository: SourceStatementRepository,
    ) -> None:
        self._repository = repository

    async def run(self, state: SourceSemanticState) -> dict[str, int]:
        await self._repository.replace_source_statement_hubs(
            state.source_uuid,
            state.source_statement_hubs,
            state.source_statement_hub_memberships,
        )
        return {'source_statement_hub_count': len(state.source_statement_hubs)}
