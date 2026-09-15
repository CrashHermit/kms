"""Persist typed predicate descriptions without changing graph topology."""

from collections.abc import Awaitable, Callable

from kms2.database.semantic.repository import SemanticRepository
from kms2.langgraph.semantic.state import SemanticState


class SourcePredicatePersistenceNode:
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
