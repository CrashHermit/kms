"""Persist typed source procedure descriptions."""

from kms2.database.semantic.source_procedure_repository import (
    SourceProcedureRepository,
)
from kms2.langgraph.semantic.state import SemanticState


class SourceProcedurePersistenceNode:
    """Update descriptions and embeddings on source procedure nodes."""

    def __init__(
        self,
        repository: SourceProcedureRepository,
    ) -> None:
        self._repository = repository

    async def run(self, state: SemanticState) -> dict[str, int]:
        """Persist all embedded source procedures and return their count."""
        if state.source_procedure_embedding_results:
            await self._repository.update_source_procedure_description(
                state.source_uuid,
                state.source_procedure_embedding_results,
            )
        return {
            'source_procedure_description_persisted_count': len(
                state.source_procedure_embedding_results
            )
        }
