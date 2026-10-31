"""Persist proposition embeddings after inference."""

from kms2.database.source_semantic.source_proposition_repository import (
    SourcePropositionRepository,
)
from kms2.langgraph.source_semantic.state import SourceSemanticState


class SourcePropositionEmbeddingPersistenceNode:
    """Update embeddings for the persisted proposition occurrences."""

    def __init__(self, repository: SourcePropositionRepository) -> None:
        self._repository = repository

    async def run(self, state: SourceSemanticState) -> dict[str, int]:
        """Persist embedding results when there are any."""
        results = state.source_proposition_embedding_results
        if results:
            await self._repository.update_source_proposition_embeddings(
                state.source_uuid, results
            )
        return {'source_proposition_embedding_persisted_count': len(results)}
