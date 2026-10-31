"""Persistence access for two-pass source-learning artifacts."""

from collections.abc import Callable, Sequence

from neo4j import AsyncSession

from kms2.core.model.source_extraction import (
    SourceEvidenceInput,
)
from kms2.core.model.source_learning.atomic import SourceAtomicFlashcardRequest
from kms2.core.model.source_learning.context import (
    SourceLearningEvidence,
    SourceLearningHubContext,
)
from kms2.core.model.source_learning.flashcard import SourceFlashcardOccurrence
from kms2.database.source_learning.queries.source_flashcard import (
    CREATE_SOURCE_FLASHCARDS,
)
from kms2.database.source_learning.queries.source_learning import (
    CLEAR_SOURCE_LEARNING,
)
from kms2.database.source_learning.queries.source_proposition import (
    READ_SOURCE_ATOMIC_FLASHCARD_REQUESTS,
)


class SourceLearningRepository:
    """Load source-learning requests and persist generic cards."""

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

    async def load_atomic_flashcard_requests(
        self, source_uuid: str
    ) -> list[SourceAtomicFlashcardRequest]:
        """Load ordered proposition requests with owning-block evidence."""
        async with self._session_factory() as session:
            result = await session.run(
                READ_SOURCE_ATOMIC_FLASHCARD_REQUESTS,
                source_uuid=source_uuid,
            )
            rows = await result.data()

        contexts: dict[str, SourceLearningHubContext] = {}
        requests: list[SourceAtomicFlashcardRequest] = []
        for row in rows:
            context = contexts.setdefault(
                row['hub_uuid'],
                SourceLearningHubContext(
                    proposition_statement=row['proposition_statement'],
                ),
            )
            source_context = SourceEvidenceInput(
                target_blocks=[
                    {
                        'block_type': row['block_type'],
                        'content': row['content'],
                    }
                ],
            )
            requests.append(
                SourceAtomicFlashcardRequest(
                    source_block_uuid=row['source_block_uuid'],
                    hub_uuid=row['hub_uuid'],
                    source_proposition_uuid=row['source_proposition_uuid'],
                    input={
                        'context': context,
                        'evidence': SourceLearningEvidence(
                            proposition_text=row['proposition_text'],
                            source_context=source_context,
                        ),
                    },
                )
            )
        return requests

    async def persist_flashcards(
        self,
        source_uuid: str,
        occurrences: Sequence[SourceFlashcardOccurrence],
    ) -> int:
        """Persist cards without replacing collection-owned identities."""
        rows = [
            {
                'uuid': occurrence.card.uuid,
                'question': occurrence.card.question,
                'answer': occurrence.card.answer,
                'source_proposition_uuids': occurrence.source_proposition_uuids,
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
