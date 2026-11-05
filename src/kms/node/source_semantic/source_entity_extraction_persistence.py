"""Persist collected source entity occurrences."""

from kms.database.source.source_entity_repository import (
    SourceEntityRepository,
)
from kms.langgraph.source_semantic.state import SourceSemanticState


class SourceEntityExtractionPersistenceNode:
    """Replace a source's raw entity occurrences."""

    def __init__(self, repository: SourceEntityRepository) -> None:
        self._repository = repository

    async def run(self, state: SourceSemanticState) -> dict[str, int]:
        """Persist all collected entities, including an empty result."""
        await self._repository.replace_source_entities(
            state.source_uuid, state.source_entities
        )
        return {'source_entity_persisted_count': len(state.source_entities)}
