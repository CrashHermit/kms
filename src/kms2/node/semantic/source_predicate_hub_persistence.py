"""Persist prepared source-local predicate hubs."""

from collections.abc import Awaitable, Callable

from kms2.database.semantic.repository import SemanticRepository
from kms2.langgraph.semantic.state import SemanticState


class SourcePredicateHubPersistenceNode:
    """Persist source predicate hubs independently from other hub types."""

    def __init__(
        self,
        repository: SemanticRepository,
        schema_initializer: Callable[[], Awaitable[None]],
    ) -> None:
        self._repository = repository
        self._schema_initializer = schema_initializer

    async def run(self, state: SemanticState) -> dict[str, int]:
        await self._schema_initializer()
        await self._repository.replace_source_predicate_hubs(
            state.source_uuid,
            state.source_predicate_hubs,
            state.source_predicate_hub_memberships,
        )
        return {'source_predicate_hub_count': len(state.source_predicate_hubs)}
