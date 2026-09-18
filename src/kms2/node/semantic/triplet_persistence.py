"""LangGraph node for triplet extraction persistence."""

from kms2.database.semantic.source_triplet_repository import (
    SourceTripletRepository,
)
from kms2.langgraph.semantic.state import SemanticState


class TripletPersistenceNode:
    """Persist source facts and their decomposed triplet occurrences."""

    def __init__(
        self,
        repository: SourceTripletRepository,
    ) -> None:
        self._repository = repository

    async def run(self, state: SemanticState) -> dict[str, object]:
        """Replace source facts and their decomposed triplet occurrences."""
        await self._repository.replace_source_facts_and_triplets(
            state.source_uuid,
            state.source_facts,
            state.triplet_occurrences,
        )
        return {}
