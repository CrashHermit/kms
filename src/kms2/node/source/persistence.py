"""LangGraph node for KMS2 source persistence."""

from collections.abc import Awaitable, Callable

from kms2.database.source.repository import SourceRepository
from kms2.langgraph.source.state import SourceState


class SourcePersistenceNode:
    """Persist the final source structure as a terminal graph side effect."""

    def __init__(
        self,
        repository: SourceRepository,
        schema_initializer: Callable[[], Awaitable[None]],
    ) -> None:
        self._repository = repository
        self._schema_initializer = schema_initializer

    async def run(self, state: SourceState) -> dict[str, object]:
        """Initialize the schema and replace the final embedded source graph."""
        await self._schema_initializer()
        await self._repository.replace_source(
            state.source, state.embedded_pages
        )
        return {}
