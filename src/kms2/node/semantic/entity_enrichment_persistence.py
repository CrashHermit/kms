"""Persist typed entity enrichment without changing graph topology."""

from collections.abc import Awaitable, Callable

from kms2.database.semantic.repository import SemanticRepository
from kms2.langgraph.semantic.state import SemanticState


class EntityEnrichmentPersistenceNode:
    """Update descriptions and embeddings on existing entity nodes."""

    def __init__(
        self,
        repository: SemanticRepository,
        schema_initializer: Callable[[], Awaitable[None]],
    ) -> None:
        self._repository = repository
        self._schema_initializer = schema_initializer

    async def run(self, state: SemanticState) -> dict[str, int]:
        """Persist all embedded entities and return their count."""
        await self._schema_initializer()
        if state.entity_embedding_results:
            await self._repository.update_entity_enrichment(
                state.source_uuid,
                state.entity_embedding_results,
            )
        return {'entity_persisted_count': len(state.entity_embedding_results)}
