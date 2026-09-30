"""Persistence access for source-learning evidence and generated artifacts."""

from collections.abc import Callable
from typing import Any

from kms2.core.model.source_learning.entity import (
    SourceEntityFlashcardInput,
    SourceEntityFlashcardResult,
    SourceEntityLearningFact,
    SourceEntityLearningFactInput,
    SourceEntityLearningFactResult,
)
from kms2.core.model.source_learning.event import (
    SourceEventFlashcardInput,
    SourceEventFlashcardResult,
    SourceEventLearningFact,
    SourceEventLearningFactInput,
    SourceEventLearningFactResult,
)
from kms2.core.model.source_learning.flashcard import SourceFlashcard
from kms2.core.model.source_learning.predicate import (
    SourcePredicateFlashcardInput,
    SourcePredicateFlashcardResult,
    SourcePredicateLearningFact,
    SourcePredicateLearningFactInput,
    SourcePredicateLearningFactResult,
)
from kms2.core.model.source_learning.triplet import (
    SourceTripletFlashcardInput,
    SourceTripletFlashcardResult,
    SourceTripletLearningFact,
    SourceTripletLearningFactInput,
    SourceTripletLearningFactResult,
)
from kms2.database.source_learning.queries.source_entity import (
    CREATE_SOURCE_ENTITY_LEARNING_FACTS,
    READ_SOURCE_ENTITY_FLASHCARD_INPUTS,
    READ_SOURCE_ENTITY_LEARNING_INPUTS,
)
from kms2.database.source_learning.queries.source_event import (
    CREATE_SOURCE_EVENT_LEARNING_FACTS,
    READ_SOURCE_EVENT_FLASHCARD_INPUTS,
    READ_SOURCE_EVENT_LEARNING_INPUTS,
)
from kms2.database.source_learning.queries.source_flashcard import (
    CREATE_SOURCE_FLASHCARDS,
)
from kms2.database.source_learning.queries.source_learning import (
    CLEAR_SOURCE_LEARNING,
)
from kms2.database.source_learning.queries.source_predicate import (
    CREATE_SOURCE_PREDICATE_LEARNING_FACTS,
    READ_SOURCE_PREDICATE_FLASHCARD_INPUTS,
    READ_SOURCE_PREDICATE_LEARNING_INPUTS,
)
from kms2.database.source_learning.queries.source_triplet import (
    CREATE_SOURCE_TRIPLET_LEARNING_FACTS,
    READ_SOURCE_TRIPLET_FLASHCARD_INPUTS,
    READ_SOURCE_TRIPLET_LEARNING_INPUTS,
)


class SourceLearningRepository:
    """Read source-local hub evidence and replace learning artifacts."""

    def __init__(self, session_factory: Callable[..., Any]) -> None:
        self._session_factory = session_factory

    async def clear_source_learning(self, source_uuid: str) -> None:
        """Delete one source's facts, cards, shared reviews, and history."""
        async with self._session_factory() as session:
            result = await session.run(
                CLEAR_SOURCE_LEARNING, source_uuid=source_uuid
            )
            await result.consume()

    async def load_entity_learning_inputs(
        self, source_uuid: str
    ) -> list[SourceEntityLearningFactInput]:
        """Load source-local entity hub evidence."""
        return await self._load_inputs(
            READ_SOURCE_ENTITY_LEARNING_INPUTS,
            SourceEntityLearningFactInput,
            source_uuid,
        )

    async def load_event_learning_inputs(
        self, source_uuid: str
    ) -> list[SourceEventLearningFactInput]:
        """Load source-local event hub evidence."""
        return await self._load_inputs(
            READ_SOURCE_EVENT_LEARNING_INPUTS,
            SourceEventLearningFactInput,
            source_uuid,
        )

    async def load_predicate_learning_inputs(
        self, source_uuid: str
    ) -> list[SourcePredicateLearningFactInput]:
        """Load source-local predicate hub evidence."""
        return await self._load_inputs(
            READ_SOURCE_PREDICATE_LEARNING_INPUTS,
            SourcePredicateLearningFactInput,
            source_uuid,
        )

    async def load_triplet_learning_inputs(
        self, source_uuid: str
    ) -> list[SourceTripletLearningFactInput]:
        """Load source-local triplet hub evidence."""
        return await self._load_inputs(
            READ_SOURCE_TRIPLET_LEARNING_INPUTS,
            SourceTripletLearningFactInput,
            source_uuid,
        )

    async def persist_entity_learning_facts(
        self,
        source_uuid: str,
        inputs: list[SourceEntityLearningFactInput],
        results: list[SourceEntityLearningFactResult],
    ) -> int:
        """Persist one learning-fact node per row with selected evidence."""
        return await self._persist_learning_facts(
            source_uuid,
            inputs,
            results,
            CREATE_SOURCE_ENTITY_LEARNING_FACTS,
            SourceEntityLearningFact,
        )

    async def persist_event_learning_facts(
        self,
        source_uuid: str,
        inputs: list[SourceEventLearningFactInput],
        results: list[SourceEventLearningFactResult],
    ) -> int:
        """Persist one learning-fact node per row with selected evidence."""
        return await self._persist_learning_facts(
            source_uuid,
            inputs,
            results,
            CREATE_SOURCE_EVENT_LEARNING_FACTS,
            SourceEventLearningFact,
        )

    async def persist_predicate_learning_facts(
        self,
        source_uuid: str,
        inputs: list[SourcePredicateLearningFactInput],
        results: list[SourcePredicateLearningFactResult],
    ) -> int:
        """Persist one learning-fact node per row with selected evidence."""
        return await self._persist_learning_facts(
            source_uuid,
            inputs,
            results,
            CREATE_SOURCE_PREDICATE_LEARNING_FACTS,
            SourcePredicateLearningFact,
        )

    async def persist_triplet_learning_facts(
        self,
        source_uuid: str,
        inputs: list[SourceTripletLearningFactInput],
        results: list[SourceTripletLearningFactResult],
    ) -> int:
        """Persist one learning-fact node per row with selected evidence."""
        return await self._persist_learning_facts(
            source_uuid,
            inputs,
            results,
            CREATE_SOURCE_TRIPLET_LEARNING_FACTS,
            SourceTripletLearningFact,
        )

    async def load_entity_flashcard_inputs(
        self, source_uuid: str
    ) -> list[SourceEntityFlashcardInput]:
        """Load persisted entity learning facts for card creation."""
        return await self._load_card_inputs(
            source_uuid,
            READ_SOURCE_ENTITY_FLASHCARD_INPUTS,
            SourceEntityFlashcardInput,
            SourceEntityLearningFact,
        )

    async def load_event_flashcard_inputs(
        self, source_uuid: str
    ) -> list[SourceEventFlashcardInput]:
        """Load persisted event learning facts for card creation."""
        return await self._load_card_inputs(
            source_uuid,
            READ_SOURCE_EVENT_FLASHCARD_INPUTS,
            SourceEventFlashcardInput,
            SourceEventLearningFact,
        )

    async def load_predicate_flashcard_inputs(
        self, source_uuid: str
    ) -> list[SourcePredicateFlashcardInput]:
        """Load persisted predicate learning facts for card creation."""
        return await self._load_card_inputs(
            source_uuid,
            READ_SOURCE_PREDICATE_FLASHCARD_INPUTS,
            SourcePredicateFlashcardInput,
            SourcePredicateLearningFact,
        )

    async def load_triplet_flashcard_inputs(
        self, source_uuid: str
    ) -> list[SourceTripletFlashcardInput]:
        """Load persisted triplet learning facts for card creation."""
        return await self._load_card_inputs(
            source_uuid,
            READ_SOURCE_TRIPLET_FLASHCARD_INPUTS,
            SourceTripletFlashcardInput,
            SourceTripletLearningFact,
        )

    async def persist_entity_flashcards(
        self,
        source_uuid: str,
        inputs: list[SourceEntityFlashcardInput],
        results: list[SourceEntityFlashcardResult],
    ) -> int:
        """Persist one entity card for every entity learning fact result."""
        return await self._persist_flashcards(source_uuid, inputs, results)

    async def persist_event_flashcards(
        self,
        source_uuid: str,
        inputs: list[SourceEventFlashcardInput],
        results: list[SourceEventFlashcardResult],
    ) -> int:
        """Persist one event card for every event learning fact result."""
        return await self._persist_flashcards(source_uuid, inputs, results)

    async def persist_predicate_flashcards(
        self,
        source_uuid: str,
        inputs: list[SourcePredicateFlashcardInput],
        results: list[SourcePredicateFlashcardResult],
    ) -> int:
        """Persist one predicate card for every predicate learning fact result."""
        return await self._persist_flashcards(source_uuid, inputs, results)

    async def persist_triplet_flashcards(
        self,
        source_uuid: str,
        inputs: list[SourceTripletFlashcardInput],
        results: list[SourceTripletFlashcardResult],
    ) -> int:
        """Persist one triplet card for every triplet learning fact result."""
        return await self._persist_flashcards(source_uuid, inputs, results)

    async def _load_inputs(
        self,
        query: str,
        model: type,
        source_uuid: str,
    ) -> list:
        async with self._session_factory() as session:
            result = await session.run(query, source_uuid=source_uuid)
            rows = await result.data()
        return [model.model_validate(row) for row in rows]

    async def _persist_learning_facts(
        self,
        source_uuid: str,
        inputs: list,
        results: list,
        query: str,
        learning_fact_model: type,
    ) -> int:
        rows: list[dict[str, object]] = []
        for learning_input, result in zip(inputs, results, strict=True):
            for fact in result.facts:
                learning_fact = learning_fact_model(
                    text=fact.text,
                )
                rows.append(
                    {
                        'uuid': learning_fact.uuid,
                        'text': learning_fact.text,
                        'hub_uuid': learning_input.hub_uuid,
                        'source_fact_uuids': fact.source_fact_uuids,
                        'triplet_uuids': fact.triplet_uuids,
                    }
                )
        if not rows:
            return 0
        async with self._session_factory() as session:
            response = await session.run(
                query,
                source_uuid=source_uuid,
                rows=rows,
            )
            record = await response.single(strict=True)
        return int(record['persisted'])

    async def _load_card_inputs(
        self,
        source_uuid: str,
        query: str,
        model: type,
        learning_fact_model: type,
    ) -> list:
        async with self._session_factory() as session:
            result = await session.run(query, source_uuid=source_uuid)
            rows = await result.data()
        return [
            model(
                learning_fact=learning_fact_model(
                    uuid=row['learning_fact_uuid'],
                    text=row['learning_fact_text'],
                ),
                learning_fact_text=row['learning_fact_text'],
                hub_uuid=row['hub_uuid'],
                hub_name=row['hub_name'],
                hub_description=row['hub_description'],
                evidence=row['evidence'],
            )
            for row in rows
        ]

    async def _persist_flashcards(
        self,
        source_uuid: str,
        inputs: list,
        results: list,
    ) -> int:
        rows = [
            {
                'uuid': SourceFlashcard(
                    question=result.question,
                    answer=result.answer,
                ).uuid,
                'question': result.question,
                'answer': result.answer,
                'learning_fact_uuid': learning_input.learning_fact.uuid,
            }
            for learning_input, result in zip(inputs, results, strict=True)
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
