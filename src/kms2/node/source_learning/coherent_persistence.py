"""Persist prepared coherent source flashcards."""

from kms2.database.source_learning.repository import SourceLearningRepository
from kms2.langgraph.source_learning.state import SourceLearningState


class SourceCoherentFlashcardPersistenceNode:
    """Persist coherent cards after their atomic parents exist."""

    def __init__(self, repository: SourceLearningRepository) -> None:
        self._repository = repository

    async def run(self, state: SourceLearningState) -> dict[str, int]:
        """Persist collected coherent occurrences with selected provenance."""
        count = await self._repository.persist_flashcards(
            state.source_uuid,
            state.coherent_occurrences,
        )
        return {'coherent_flashcard_count': count}
