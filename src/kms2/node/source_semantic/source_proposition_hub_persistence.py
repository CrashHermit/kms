"""Persist prepared source-local proposition hubs."""

from kms2.database.source_semantic.source_proposition_repository import (
    SourcePropositionRepository,
)
from kms2.langgraph.source_semantic.state import SourceSemanticState


class SourcePropositionHubPersistenceNode:
    """Persist proposition hubs and their occurrence memberships."""

    def __init__(self, repository: SourcePropositionRepository) -> None:
        self._repository = repository

    async def run(self, state: SourceSemanticState) -> dict[str, int]:
        await self._repository.replace_source_proposition_hubs(
            state.source_uuid,
            state.source_proposition_hubs,
            state.source_proposition_hub_memberships,
        )
        return {
            'source_proposition_hub_count': len(state.source_proposition_hubs)
        }
