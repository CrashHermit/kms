"""LangGraph node for loading persisted facts into triplet decomposition."""

from kms2.database.source_semantic.source_fact_repository import (
    SourceFactRepository,
)
from kms2.langgraph.source_semantic.state import SourceSemanticState


class SourceTripletFactLoadNode:
    """Load the durable source facts consumed by triplet decomposition."""

    def __init__(self, repository: SourceFactRepository) -> None:
        self._repository = repository

    async def run(self, state: SourceSemanticState) -> dict[str, object]:
        """Read facts after the independent fact persistence phase."""
        return {
            'source_facts': await self._repository.load_source_facts(
                state.source_uuid
            )
        }
