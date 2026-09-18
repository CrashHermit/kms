"""Persist typed source statement descriptions."""

from kms2.database.semantic.source_statement_repository import (
    SourceStatementRepository,
)
from kms2.langgraph.semantic.state import SemanticState


class SourceStatementPersistenceNode:
    """Update descriptions and embeddings on source statement nodes."""

    def __init__(
        self,
        repository: SourceStatementRepository,
    ) -> None:
        self._repository = repository

    async def run(self, state: SemanticState) -> dict[str, int]:
        """Persist all embedded source statements and return their count."""
        if state.source_statement_embedding_results:
            await self._repository.update_source_statement_description(
                state.source_uuid,
                state.source_statement_embedding_results,
            )
        return {
            'source_statement_description_persisted_count': len(
                state.source_statement_embedding_results
            )
        }
