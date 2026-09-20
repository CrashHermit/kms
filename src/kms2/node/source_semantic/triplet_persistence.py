"""LangGraph node for triplet extraction persistence."""

from kms2.database.source_semantic.source_triplet_repository import (
    SourceTripletRepository,
)
from kms2.langgraph.source_semantic.state import SourceSemanticState


class TripletPersistenceNode:
    """Persist source facts and their decomposed triplet occurrences."""

    def __init__(
        self,
        repository: SourceTripletRepository,
    ) -> None:
        self._repository = repository

    async def run(self, state: SourceSemanticState) -> dict[str, object]:
        """Replace source facts and their decomposed triplet occurrences."""
        await self._repository.replace_source_facts_and_triplets(
            state.source_uuid,
            state.source_facts,
            state.triplet_occurrences,
        )
        return {}
