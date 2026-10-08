"""Persistence access for two-pass source-learning artifacts."""

from collections.abc import Callable, Sequence

from neo4j import AsyncSession

from kms2.core.model.source_learning.atomic import SourceAtomicFlashcardRequest
from kms2.core.model.source_learning.context import (
    SourceLearningEvidence,
    SourceLearningHubContext,
)
from kms2.core.model.source_learning.flashcard import SourceFlashcardOccurrence
from kms2.core.model.source_semantic.source_triplet_hub import (
    SourceTripletHubRole,
)
from kms2.database.source_learning.queries.source_flashcard import (
    CREATE_SOURCE_FLASHCARDS,
)
from kms2.database.source_learning.queries.source_learning import (
    CLEAR_SOURCE_LEARNING,
)
from kms2.database.source_learning.queries.source_triplet import (
    READ_SOURCE_ATOMIC_FLASHCARD_REQUESTS,
)


class SourceLearningRepository:
    """Read exact source-semantic context and persist generic cards."""

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
        """Load ordered exact triplet/fact requests with shared hub context."""
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
                    triplet_hub=SourceTripletHubRole(
                        name=row['triplet_hub_name'],
                        description=row['triplet_hub_description'],
                    ),
                    subject_hub=SourceTripletHubRole(
                        name=row['subject_hub_name'],
                        description=row['subject_hub_description'],
                    ),
                    predicate_hub=SourceTripletHubRole(
                        name=row['predicate_hub_name'],
                        description=row['predicate_hub_description'],
                    ),
                    object_hub=SourceTripletHubRole(
                        name=row['object_hub_name'],
                        description=row['object_hub_description'],
                    ),
                ),
            )
            requests.append(
                SourceAtomicFlashcardRequest(
                    hub_uuid=row['hub_uuid'],
                    triplet_uuid=row['triplet_uuid'],
                    source_fact_uuid=row['source_fact_uuid'],
                    input={
                        'context': context,
                        'evidence': SourceLearningEvidence(
                            subject=row['subject'],
                            predicate=row['predicate'],
                            object=row['object'],
                            source_fact_text=row['source_fact_text'],
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
                'hub_uuid': occurrence.hub_uuid,
                'triplet_uuids': occurrence.triplet_uuids,
                'source_fact_uuids': occurrence.source_fact_uuids,
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
