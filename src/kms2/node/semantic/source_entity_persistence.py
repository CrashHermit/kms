"""Persist typed entity descriptions without changing graph topology."""

from kms2.database.semantic.source_entity_repository import (
    SourceEntityRepository,
)
from kms2.langgraph.semantic.state import SemanticState


class SourceEntityPersistenceNode:
    """Update descriptions and embeddings on existing entity nodes."""

    def __init__(
        self,
        repository: SourceEntityRepository,
    ) -> None:
        self._repository = repository

    async def run(self, state: SemanticState) -> dict[str, int]:
        """Persist all embedded entities and return their count."""
        if state.source_entity_embedding_results:
            await self._repository.update_source_entity_description(
                state.source_uuid,
                state.source_entity_embedding_results,
            )
        return {
            'source_entity_description_persisted_count': len(
                state.source_entity_embedding_results
            )
        }
