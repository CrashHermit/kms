"""LangGraph node for triplet extraction persistence."""

from kms2.database.source_semantic.source_triplet_repository import (
    SourceTripletRepository,
)
from kms2.langgraph.source_semantic.state import SourceSemanticState


class SourceTripletPersistenceNode:
    """Persist source triplets without replacing durable source facts."""

    def __init__(
        self,
        repository: SourceTripletRepository,
    ) -> None:
        self._repository = repository

    async def run(self, state: SourceSemanticState) -> dict[str, object]:
        """Replace only the decomposed triplet layer."""
        await self._repository.replace_source_triplets(
            state.source_uuid,
            state.triplet_occurrences,
        )
        return {}
