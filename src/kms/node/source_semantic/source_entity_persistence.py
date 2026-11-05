"""Persist typed entity descriptions without changing graph topology."""

from kms.database.source.source_entity_repository import (
    SourceEntityRepository,
)
from kms.langgraph.source_semantic.state import SourceSemanticState


class SourceEntityPersistenceNode:
    """Update descriptions and embeddings on existing entity nodes."""

    def __init__(
        self,
        repository: SourceEntityRepository,
    ) -> None:
        self._repository = repository

    async def run(self, state: SourceSemanticState) -> dict[str, int]:
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
