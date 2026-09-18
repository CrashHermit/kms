"""Persist typed event descriptions without changing graph topology."""

from kms2.database.semantic.source_event_repository import SourceEventRepository
from kms2.langgraph.semantic.state import SemanticState


class SourceEventPersistenceNode:
    """Update descriptions and embeddings on existing event nodes."""

    def __init__(
        self,
        repository: SourceEventRepository,
    ) -> None:
        self._repository = repository

    async def run(self, state: SemanticState) -> dict[str, int]:
        """Persist all embedded events and return their count."""
        if state.source_event_embedding_results:
            await self._repository.update_source_event_description(
                state.source_uuid,
                state.source_event_embedding_results,
            )
        return {
            'source_event_description_persisted_count': len(
                state.source_event_embedding_results
            )
        }
