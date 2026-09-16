"""Persist prepared source-local statement hubs."""

from collections.abc import Awaitable, Callable

from kms2.database.semantic.repository import SemanticRepository
from kms2.langgraph.semantic.state import SemanticState


class SourceStatementHubPersistenceNode:
    """Persist source statement hubs independently from other hub types."""

    def __init__(
        self,
        repository: SemanticRepository,
        schema_initializer: Callable[[], Awaitable[None]],
    ) -> None:
        self._repository = repository
        self._schema_initializer = schema_initializer

    async def run(self, state: SemanticState) -> dict[str, int]:
        await self._schema_initializer()
        await self._repository.replace_source_statement_hubs(
            state.source_uuid,
            state.source_statement_hubs,
            state.source_statement_hub_memberships,
        )
        return {'source_statement_hub_count': len(state.source_statement_hubs)}
