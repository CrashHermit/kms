"""Persist prepared atomic source flashcards."""

from kms2.database.source_learning.repository import SourceLearningRepository
from kms2.langgraph.source_learning.state import SourceLearningState


class SourceAtomicFlashcardPersistenceNode:
    """Persist atomic cards before coherent derivation begins."""

    def __init__(self, repository: SourceLearningRepository) -> None:
        self._repository = repository

    async def run(self, state: SourceLearningState) -> dict[str, int]:
        """Persist collected atomic occurrences without replacing identities."""
        occurrences = [
            occurrence
            for packet in state.atomic_packets
            for occurrence in packet.cards
        ]
        count = await self._repository.persist_flashcards(
            state.source_uuid,
            occurrences,
        )
        return {'atomic_flashcard_count': count}
