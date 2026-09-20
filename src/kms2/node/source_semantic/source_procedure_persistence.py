"""Persist typed source procedure descriptions."""

from kms2.database.source_semantic.source_procedure_repository import (
    SourceProcedureRepository,
)
from kms2.langgraph.source_semantic.state import SourceSemanticState


class SourceProcedurePersistenceNode:
    """Update descriptions and embeddings on source procedure nodes."""

    def __init__(
        self,
        repository: SourceProcedureRepository,
    ) -> None:
        self._repository = repository

    async def run(self, state: SourceSemanticState) -> dict[str, int]:
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
