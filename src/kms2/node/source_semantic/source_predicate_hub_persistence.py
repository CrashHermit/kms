"""Persist prepared source-local predicate hubs."""

from kms2.database.source_semantic.source_predicate_repository import (
    SourcePredicateRepository,
)
from kms2.langgraph.source_semantic.state import SourceSemanticState


class SourcePredicateHubPersistenceNode:
    """Persist source predicate hubs independently from other hub types."""

    def __init__(
        self,
        repository: SourcePredicateRepository,
    ) -> None:
        self._repository = repository

    async def run(self, state: SourceSemanticState) -> dict[str, int]:
        await self._repository.replace_source_predicate_hubs(
            state.source_uuid,
            state.source_predicate_hubs,
            state.source_predicate_hub_memberships,
        )
        return {'source_predicate_hub_count': len(state.source_predicate_hubs)}
