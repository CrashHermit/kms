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
from kms2.database.source_learning.queries import (
    _CLEAR_SOURCE_LEARNING,
    _READ_LEARNING_FACTS,
    PERSIST_SOURCE_FLASHCARDS,
    READ_SOURCE_ENTITY_LEARNING_INPUTS,
    READ_SOURCE_EVENT_LEARNING_INPUTS,
    READ_SOURCE_PREDICATE_LEARNING_INPUTS,
    READ_SOURCE_TRIPLET_LEARNING_INPUTS,
    REPLACE_SOURCE_LEARNING_FACTS,
)


class SourceLearningRepository:
    """Read source-local hub evidence and replace learning artifacts."""

    def __init__(self, session_factory: Callable[..., Any]) -> None:
        self._session_factory = session_factory

    async def clear_source_learning(self, source_uuid: str) -> None:
        """Delete generated learning records for one source only."""
        async with self._session_factory() as session:
            result = await session.run(
                _CLEAR_SOURCE_LEARNING, source_uuid=source_uuid
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

    async def replace_entity_learning_facts(
        self,
        source_uuid: str,
        inputs: list[SourceEntityLearningFactInput],
        results: list[SourceEntityLearningFactResult],
    ) -> int:
        """Persist entity learning facts and their selected provenance."""
        return await self._replace_learning_facts(
            source_uuid,
            inputs,
            results,
            REPLACE_SOURCE_LEARNING_FACTS['entity'],
            SourceEntityLearningFact,
        )

    async def replace_event_learning_facts(
        self,
        source_uuid: str,
        inputs: list[SourceEventLearningFactInput],
        results: list[SourceEventLearningFactResult],
    ) -> int:
        """Persist event learning facts and their selected provenance."""
        return await self._replace_learning_facts(
            source_uuid,
            inputs,
            results,
            REPLACE_SOURCE_LEARNING_FACTS['event'],
            SourceEventLearningFact,
        )

    async def replace_predicate_learning_facts(
        self,
        source_uuid: str,
        inputs: list[SourcePredicateLearningFactInput],
        results: list[SourcePredicateLearningFactResult],
    ) -> int:
        """Persist predicate learning facts and their selected provenance."""
        return await self._replace_learning_facts(
            source_uuid,
            inputs,
            results,
            REPLACE_SOURCE_LEARNING_FACTS['predicate'],
            SourcePredicateLearningFact,
        )

    async def replace_triplet_learning_facts(
        self,
        source_uuid: str,
        inputs: list[SourceTripletLearningFactInput],
        results: list[SourceTripletLearningFactResult],
    ) -> int:
        """Persist triplet learning facts and their selected provenance."""
        return await self._replace_learning_facts(
            source_uuid,
            inputs,
            results,
            REPLACE_SOURCE_LEARNING_FACTS['triplet'],
            SourceTripletLearningFact,
        )

    async def load_entity_flashcard_inputs(
        self, source_uuid: str
    ) -> list[SourceEntityFlashcardInput]:
        """Load persisted entity learning facts for card creation."""
        return await self._load_card_inputs(
            source_uuid,
            _READ_LEARNING_FACTS['entity'],
            SourceEntityFlashcardInput,
            SourceEntityLearningFact,
        )

    async def load_event_flashcard_inputs(
        self, source_uuid: str
    ) -> list[SourceEventFlashcardInput]:
        """Load persisted event learning facts for card creation."""
        return await self._load_card_inputs(
            source_uuid,
            _READ_LEARNING_FACTS['event'],
            SourceEventFlashcardInput,
            SourceEventLearningFact,
        )

    async def load_predicate_flashcard_inputs(
        self, source_uuid: str
    ) -> list[SourcePredicateFlashcardInput]:
        """Load persisted predicate learning facts for card creation."""
        return await self._load_card_inputs(
            source_uuid,
            _READ_LEARNING_FACTS['predicate'],
            SourcePredicateFlashcardInput,
            SourcePredicateLearningFact,
        )

    async def load_triplet_flashcard_inputs(
        self, source_uuid: str
    ) -> list[SourceTripletFlashcardInput]:
        """Load persisted triplet learning facts for card creation."""
        return await self._load_card_inputs(
            source_uuid,
            _READ_LEARNING_FACTS['triplet'],
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

    async def _replace_learning_facts(
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
            record = await response.single()
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
                PERSIST_SOURCE_FLASHCARDS,
                source_uuid=source_uuid,
                rows=rows,
            )
            record = await response.single()
        return int(record['persisted'])
