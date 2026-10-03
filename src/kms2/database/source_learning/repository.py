"""Persistence access for source-learning requests and generated artifacts."""

from collections.abc import Callable, Sequence

from neo4j import AsyncSession
from pydantic import BaseModel

from kms2.core.model.source_learning.entity import (
    SourceEntityFlashcardRequest,
    SourceEntityLearningFact,
    SourceEntityLearningFactOccurrence,
    SourceEntityLearningFactRequest,
)
from kms2.core.model.source_learning.event import (
    SourceEventFlashcardRequest,
    SourceEventLearningFact,
    SourceEventLearningFactOccurrence,
    SourceEventLearningFactRequest,
)
from kms2.core.model.source_learning.flashcard import (
    SourceFlashcardOccurrence,
)
from kms2.core.model.source_learning.predicate import (
    SourcePredicateFlashcardRequest,
    SourcePredicateLearningFact,
    SourcePredicateLearningFactOccurrence,
    SourcePredicateLearningFactRequest,
)
from kms2.core.model.source_learning.triplet import (
    SourceTripletFlashcardRequest,
    SourceTripletLearningFact,
    SourceTripletLearningFactOccurrence,
    SourceTripletLearningFactRequest,
)
from kms2.database.source_learning.queries.source_entity import (
    CREATE_SOURCE_ENTITY_LEARNING_FACTS,
    READ_SOURCE_ENTITY_FLASHCARD_REQUESTS,
    READ_SOURCE_ENTITY_LEARNING_REQUESTS,
)
from kms2.database.source_learning.queries.source_event import (
    CREATE_SOURCE_EVENT_LEARNING_FACTS,
    READ_SOURCE_EVENT_FLASHCARD_REQUESTS,
    READ_SOURCE_EVENT_LEARNING_REQUESTS,
)
from kms2.database.source_learning.queries.source_flashcard import (
    CREATE_SOURCE_FLASHCARDS,
)
from kms2.database.source_learning.queries.source_learning import (
    CLEAR_SOURCE_LEARNING,
)
from kms2.database.source_learning.queries.source_predicate import (
    CREATE_SOURCE_PREDICATE_LEARNING_FACTS,
    READ_SOURCE_PREDICATE_FLASHCARD_REQUESTS,
    READ_SOURCE_PREDICATE_LEARNING_REQUESTS,
)
from kms2.database.source_learning.queries.source_triplet import (
    CREATE_SOURCE_TRIPLET_LEARNING_FACTS,
    READ_SOURCE_TRIPLET_FLASHCARD_REQUESTS,
    READ_SOURCE_TRIPLET_LEARNING_REQUESTS,
)

type _LearningOccurrenceModel = (
    SourceEntityLearningFactOccurrence
    | SourceEventLearningFactOccurrence
    | SourcePredicateLearningFactOccurrence
    | SourceTripletLearningFactOccurrence
)
type _LearningFactModel = (
    SourceEntityLearningFact
    | SourceEventLearningFact
    | SourcePredicateLearningFact
    | SourceTripletLearningFact
)


class SourceLearningRepository:
    """Read source-local requests and persist collection-created artifacts."""

    def __init__(self, session_factory: Callable[[], AsyncSession]) -> None:
        self._session_factory = session_factory

    async def clear_source_learning(self, source_uuid: str) -> None:
        """Delete one source's facts, cards, shared reviews, and history."""
        async with self._session_factory() as session:
            result = await session.run(
                CLEAR_SOURCE_LEARNING, source_uuid=source_uuid
            )
            await result.consume()

    async def load_entity_learning_requests(
        self, source_uuid: str
    ) -> list[SourceEntityLearningFactRequest]:
        return await self._load_requests(
            READ_SOURCE_ENTITY_LEARNING_REQUESTS,
            SourceEntityLearningFactRequest,
            source_uuid,
        )

    async def load_event_learning_requests(
        self, source_uuid: str
    ) -> list[SourceEventLearningFactRequest]:
        return await self._load_requests(
            READ_SOURCE_EVENT_LEARNING_REQUESTS,
            SourceEventLearningFactRequest,
            source_uuid,
        )

    async def load_predicate_learning_requests(
        self, source_uuid: str
    ) -> list[SourcePredicateLearningFactRequest]:
        return await self._load_requests(
            READ_SOURCE_PREDICATE_LEARNING_REQUESTS,
            SourcePredicateLearningFactRequest,
            source_uuid,
        )

    async def load_triplet_learning_requests(
        self, source_uuid: str
    ) -> list[SourceTripletLearningFactRequest]:
        return await self._load_requests(
            READ_SOURCE_TRIPLET_LEARNING_REQUESTS,
            SourceTripletLearningFactRequest,
            source_uuid,
        )

    async def persist_entity_learning_facts(
        self,
        source_uuid: str,
        occurrences: Sequence[SourceEntityLearningFactOccurrence],
    ) -> int:
        return await self._persist_learning_facts(
            source_uuid, occurrences, CREATE_SOURCE_ENTITY_LEARNING_FACTS
        )

    async def persist_event_learning_facts(
        self,
        source_uuid: str,
        occurrences: Sequence[SourceEventLearningFactOccurrence],
    ) -> int:
        return await self._persist_learning_facts(
            source_uuid, occurrences, CREATE_SOURCE_EVENT_LEARNING_FACTS
        )

    async def persist_predicate_learning_facts(
        self,
        source_uuid: str,
        occurrences: Sequence[SourcePredicateLearningFactOccurrence],
    ) -> int:
        return await self._persist_learning_facts(
            source_uuid, occurrences, CREATE_SOURCE_PREDICATE_LEARNING_FACTS
        )

    async def persist_triplet_learning_facts(
        self,
        source_uuid: str,
        occurrences: Sequence[SourceTripletLearningFactOccurrence],
    ) -> int:
        return await self._persist_learning_facts(
            source_uuid, occurrences, CREATE_SOURCE_TRIPLET_LEARNING_FACTS
        )

    async def load_entity_flashcard_requests(
        self, source_uuid: str
    ) -> list[SourceEntityFlashcardRequest]:
        return await self._load_card_requests(
            source_uuid,
            READ_SOURCE_ENTITY_FLASHCARD_REQUESTS,
            SourceEntityFlashcardRequest,
            SourceEntityLearningFact,
        )

    async def load_event_flashcard_requests(
        self, source_uuid: str
    ) -> list[SourceEventFlashcardRequest]:
        return await self._load_card_requests(
            source_uuid,
            READ_SOURCE_EVENT_FLASHCARD_REQUESTS,
            SourceEventFlashcardRequest,
            SourceEventLearningFact,
        )

    async def load_predicate_flashcard_requests(
        self, source_uuid: str
    ) -> list[SourcePredicateFlashcardRequest]:
        return await self._load_card_requests(
            source_uuid,
            READ_SOURCE_PREDICATE_FLASHCARD_REQUESTS,
            SourcePredicateFlashcardRequest,
            SourcePredicateLearningFact,
        )

    async def load_triplet_flashcard_requests(
        self, source_uuid: str
    ) -> list[SourceTripletFlashcardRequest]:
        return await self._load_card_requests(
            source_uuid,
            READ_SOURCE_TRIPLET_FLASHCARD_REQUESTS,
            SourceTripletFlashcardRequest,
            SourceTripletLearningFact,
        )

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
                'learning_fact_uuid': occurrence.learning_fact_uuid,
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

    async def _load_requests[RequestModel: BaseModel](
        self, query: str, model: type[RequestModel], source_uuid: str
    ) -> list[RequestModel]:
        async with self._session_factory() as session:
            result = await session.run(query, source_uuid=source_uuid)
            rows = await result.data()
        return [model.model_validate(row) for row in rows]

    async def _load_card_requests[CardRequestModel: BaseModel](
        self,
        source_uuid: str,
        query: str,
        model: type[CardRequestModel],
        learning_fact_model: type[_LearningFactModel],
    ) -> list[CardRequestModel]:
        async with self._session_factory() as session:
            result = await session.run(query, source_uuid=source_uuid)
            rows = await result.data()
        return [
            model(
                learning_fact=learning_fact_model(
                    uuid=row['learning_fact_uuid'],
                    text=row['learning_fact_text'],
                )
            )
            for row in rows
        ]

    async def _persist_learning_facts(
        self,
        source_uuid: str,
        occurrences: Sequence[_LearningOccurrenceModel],
        query: str,
    ) -> int:
        rows = [
            {
                'uuid': occurrence.learning_fact.uuid,
                'text': occurrence.learning_fact.text,
                'hub_uuid': occurrence.hub_uuid,
                'source_fact_uuid': occurrence.source_fact_uuid,
            }
            for occurrence in occurrences
        ]
        if not rows:
            return 0
        async with self._session_factory() as session:
            response = await session.run(
                query, source_uuid=source_uuid, rows=rows
            )
            record = await response.single(strict=True)
        return int(record['persisted'])
