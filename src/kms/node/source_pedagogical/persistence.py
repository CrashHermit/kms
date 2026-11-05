"""Persist pedagogical annotations and split source pages."""

from kms.database.source.source_graph_repository import SourceGraphRepository
from kms.langgraph.source_pedagogical.state import SourcePedagogicalState


class SourcePedagogicalPersistenceNode:
    """Replace one source's pedagogical components without rebuilding it."""

    def __init__(self, repository: SourceGraphRepository) -> None:
        self._repository = repository

    async def run(self, state: SourcePedagogicalState) -> dict[str, object]:
        """Persist split pages and their pedagogical annotations."""
        await self._repository.replace_pedagogical_material(
            state.source_uuid,
            state.embedded_pages,
            state.instructions,
            state.statements,
            state.procedures,
        )
        return {}
