"""Persist collected source proposition occurrences."""

from kms2.database.source_semantic.source_proposition_repository import (
    SourcePropositionRepository,
)
from kms2.langgraph.source_semantic.state import SourceSemanticState


class SourcePropositionPersistenceNode:
    """Replace a source's raw proposition occurrences."""

    def __init__(self, repository: SourcePropositionRepository) -> None:
        self._repository = repository

    async def run(self, state: SourceSemanticState) -> dict[str, int]:
        """Persist all collected propositions, including an empty result."""
        await self._repository.replace_source_propositions(
            state.source_uuid, state.source_propositions
        )
        return {
            'source_proposition_persisted_count': len(state.source_propositions)
        }
