"""LangGraph nodes for entity learning facts and cards."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.core.model.source_learning.entity import (
    SourceEntityFlashcardRequest,
    SourceEntityLearningFact,
    SourceEntityLearningFactOccurrence,
    SourceEntityLearningFactRequest,
)
from kms2.core.model.source_learning.flashcard import (
    SourceFlashcard,
    SourceFlashcardOccurrence,
)
from kms2.database.source_learning.repository import SourceLearningRepository
from kms2.langgraph.source_learning.state import (
    SourceEntityFlashcardWorkerResult,
    SourceEntityLearningFactWorkerResult,
    SourceLearningState,
)
from kms2.module.source_learning.entity import (
    SourceEntityFlashcardModule,
    SourceEntityLearningFactModule,
)


class EntityLearningFactWorkerState(TypedDict):
    """Backend request sent to one entity learning-fact worker."""

    ordinal: int
    request: SourceEntityLearningFactRequest


class EntityFlashcardWorkerState(TypedDict):
    """Backend request sent to one entity card worker."""

    ordinal: int
    request: SourceEntityFlashcardRequest


class SourceEntityLearningFactNode:
    """Infer entity learning facts and attach their originating provenance."""

    def __init__(
        self,
        repository: SourceLearningRepository,
        module: SourceEntityLearningFactModule,
    ) -> None:
        self._repository = repository
        self._module = module

    async def load(self, state: SourceLearningState) -> dict:
        """Load distinct entity hub/source-fact requests."""
        return {
            'entity_learning_fact_requests': await self._repository.load_entity_learning_requests(
                state.source_uuid
            )
        }

    def dispatch(
        self, state: SourceLearningState
    ) -> list[Send] | Literal['source_entity_learning_fact_collect']:
        """Dispatch one worker per request in stable order."""
        return [
            Send(
                'source_entity_learning_fact_worker',
                {'ordinal': ordinal, 'request': request},
            )
            for ordinal, request in enumerate(
                state.entity_learning_fact_requests
            )
        ] or 'source_entity_learning_fact_collect'

    async def worker(
        self, state: EntityLearningFactWorkerState
    ) -> dict[str, list[SourceEntityLearningFactWorkerResult]]:
        """Infer from content only, retaining backend request order."""
        facts = await self._module.aforward(
            request=state['request'].model_input()
        )
        return {
            'entity_learning_fact_results': [
                SourceEntityLearningFactWorkerResult(
                    ordinal=state['ordinal'], facts=facts
                )
            ]
        }

    def collect(self, state: SourceLearningState) -> dict:
        """Materialize candidates with their original hub and source fact."""
        occurrences: list[SourceEntityLearningFactOccurrence] = []
        for item in sorted(
            state.entity_learning_fact_results, key=lambda item: item.ordinal
        ):
            request = state.entity_learning_fact_requests[item.ordinal]
            for candidate in item.facts:
                occurrences.append(
                    SourceEntityLearningFactOccurrence(
                        learning_fact=SourceEntityLearningFact(
                            text=candidate.text
                        ),
                        hub_uuid=request.hub_uuid,
                        source_fact_uuid=request.source_fact_uuid,
                    )
                )
        return {'entity_learning_fact_occurrences': occurrences}

    async def persist(self, state: SourceLearningState) -> dict[str, int]:
        """Persist completed entity learning-fact occurrences."""
        count = await self._repository.persist_entity_learning_facts(
            state.source_uuid, state.entity_learning_fact_occurrences
        )
        return {'entity_learning_fact_count': count}


class SourceEntityFlashcardNode:
    """Infer entity cards and attach their persisted learning-fact identity."""

    def __init__(
        self,
        repository: SourceLearningRepository,
        module: SourceEntityFlashcardModule,
    ) -> None:
        self._repository = repository
        self._module = module

    async def load(self, state: SourceLearningState) -> dict:
        """Load persisted entity learning-fact requests."""
        return {
            'entity_flashcard_requests': await self._repository.load_entity_flashcard_requests(
                state.source_uuid
            )
        }

    def dispatch(
        self, state: SourceLearningState
    ) -> list[Send] | Literal['source_entity_flashcard_collect']:
        """Dispatch one card worker per persisted learning fact."""
        return [
            Send(
                'source_entity_flashcard_worker',
                {'ordinal': ordinal, 'request': request},
            )
            for ordinal, request in enumerate(state.entity_flashcard_requests)
        ] or 'source_entity_flashcard_collect'

    async def worker(
        self, state: EntityFlashcardWorkerState
    ) -> dict[str, list[SourceEntityFlashcardWorkerResult]]:
        """Infer one card from only the persisted learning-fact text."""
        result = await self._module.aforward(
            request=state['request'].model_input()
        )
        return {
            'entity_flashcard_results': [
                SourceEntityFlashcardWorkerResult(
                    ordinal=state['ordinal'], result=result
                )
            ]
        }

    def collect(self, state: SourceLearningState) -> dict:
        """Materialize cards with their original persisted learning fact."""
        occurrences: list[SourceFlashcardOccurrence] = []
        for item in sorted(
            state.entity_flashcard_results, key=lambda item: item.ordinal
        ):
            request = state.entity_flashcard_requests[item.ordinal]
            occurrences.append(
                SourceFlashcardOccurrence(
                    card=SourceFlashcard(
                        question=item.result.question, answer=item.result.answer
                    ),
                    learning_fact_uuid=request.learning_fact.uuid,
                )
            )
        return {'entity_flashcard_occurrences': occurrences}

    async def persist(self, state: SourceLearningState) -> dict[str, int]:
        """Persist completed cards without replacing collection-owned identities."""
        count = await self._repository.persist_flashcards(
            state.source_uuid, state.entity_flashcard_occurrences
        )
        return {'entity_flashcard_count': count}
