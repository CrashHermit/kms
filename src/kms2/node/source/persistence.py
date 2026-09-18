"""LangGraph node for KMS2 source persistence."""

from kms2.database.source.source_graph_repository import SourceGraphRepository
from kms2.langgraph.source.state import SourceState


class SourcePersistenceNode:
    """Persist the final source structure as a terminal graph side effect."""

    def __init__(
        self,
        repository: SourceGraphRepository,
    ) -> None:
        self._repository = repository

    async def run(self, state: SourceState) -> dict[str, object]:
        """Replace the final embedded source graph."""
        await self._repository.replace_source(
            state.source,
            state.embedded_pages,
            state.instructions,
            state.statements,
            state.procedures,
        )
        return {}
