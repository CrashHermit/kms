"""Persistence access for two-pass source-learning artifacts."""

from collections.abc import Callable, Sequence

from neo4j import AsyncSession

from kms.core.model.source_learning.atomic import (
    SourceAtomicFlashcardMaterial,
)
from kms.core.model.source_learning.flashcard import SourceFlashcardOccurrence
from kms.database.source_learning.queries.source_flashcard import (
    CREATE_SOURCE_FLASHCARDS,
)
from kms.database.source_learning.queries.source_learning import (
    CLEAR_SOURCE_LEARNING,
)
from kms.database.source_learning.queries.source_material import (
    READ_SOURCE_ATOMIC_FLASHCARD_REQUESTS,
)


class SourceLearningRepository:
    """Load source-learning metadata and persist generic cards."""

    def __init__(self, session_factory: Callable[[], AsyncSession]) -> None:
        self._session_factory = session_factory

    async def clear_source_learning(self, source_uuid: str) -> None:
        """Delete generated cards and linked review history for one source."""
        async with self._session_factory() as session:
            result = await session.run(
                CLEAR_SOURCE_LEARNING,
                source_uuid=source_uuid,
            )
            await result.consume()

    async def load_atomic_flashcard_materials(
        self, source_uuid: str
    ) -> list[SourceAtomicFlashcardMaterial]:
        """Load ordered metadata for textual or image-bearing source owners."""
        async with self._session_factory() as session:
            result = await session.run(
                READ_SOURCE_ATOMIC_FLASHCARD_REQUESTS,
                source_uuid=source_uuid,
            )
            rows = await result.data()
        return [
            SourceAtomicFlashcardMaterial(
                source_block_uuid=row['source_block_uuid'],
                source_page_index=row['source_page_index'],
            )
            for row in rows
        ]

    async def persist_flashcards(
        self,
        source_uuid: str,
        occurrences: Sequence[SourceFlashcardOccurrence],
    ) -> int:
        rows = [
            {
                'uuid': occurrence.card.uuid,
                'question': occurrence.card.question,
                'answer': occurrence.card.answer,
                'source_block_uuids': occurrence.source_block_uuids,
                'derived_card_uuids': occurrence.derived_card_uuids,
            }
            for occurrence in occurrences
        ]
        if not rows:
            return 0

        async with self._session_factory() as session:
            response = await session.run(
                CREATE_SOURCE_FLASHCARDS,
                source_uuid=source_uuid,
                rows=rows,
            )
            record = await response.single(strict=True)
        return int(record['persisted'])
