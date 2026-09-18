"""Persist typed predicate descriptions without changing graph topology."""

from kms2.database.semantic.source_predicate_repository import (
    SourcePredicateRepository,
)
from kms2.langgraph.semantic.state import SemanticState


class SourcePredicatePersistenceNode:
    """Update descriptions and embeddings on existing predicate nodes."""

    def __init__(
        self,
        repository: SourcePredicateRepository,
    ) -> None:
        self._repository = repository

    async def run(self, state: SemanticState) -> dict[str, int]:
        """Persist all embedded predicates and return their count."""
        if state.source_predicate_embedding_results:
            await self._repository.update_source_predicate_description(
                state.source_uuid,
                state.source_predicate_embedding_results,
            )
        return {
            'source_predicate_description_persisted_count': len(
                state.source_predicate_embedding_results
            )
        }
