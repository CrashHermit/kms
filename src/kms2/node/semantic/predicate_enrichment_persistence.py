"""Persist typed predicate enrichment without changing graph topology."""

from collections.abc import Awaitable, Callable

from kms2.database.semantic.repository import SemanticRepository
from kms2.langgraph.semantic.state import SemanticState


class PredicateEnrichmentPersistenceNode:
    """Update descriptions and embeddings on existing predicate nodes."""

    def __init__(
        self,
        repository: SemanticRepository,
        schema_initializer: Callable[[], Awaitable[None]],
    ) -> None:
        self._repository = repository
        self._schema_initializer = schema_initializer

    async def run(self, state: SemanticState) -> dict[str, int]:
        """Persist all embedded predicates and return their count."""
        await self._schema_initializer()
        if state.predicate_embedding_results:
            await self._repository.update_predicate_enrichment(
                state.source_uuid,
                state.predicate_embedding_results,
            )
        return {
            'predicate_persisted_count': len(state.predicate_embedding_results)
        }
