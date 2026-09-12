"""LangGraph node for triplet extraction persistence."""

from collections.abc import Awaitable, Callable

from kms2.database.semantic.repository import SemanticRepository
from kms2.langgraph.semantic.state import SemanticState


class TripletPersistenceNode:
    """Persist raw triplet assertions as a terminal graph side effect."""

    def __init__(
        self,
        repository: SemanticRepository,
        schema_initializer: Callable[[], Awaitable[None]],
    ) -> None:
        self._repository = repository
        self._schema_initializer = schema_initializer

    async def run(self, state: SemanticState) -> dict[str, object]:
        """Initialize structural schema and replace source assertions."""
        await self._schema_initializer()
        await self._repository.replace_source_assertions(
            state.source_uuid,
            state.raw_assertions,
        )
        return {}
