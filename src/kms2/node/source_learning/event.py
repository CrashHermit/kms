"""LangGraph nodes for event learning facts and cards."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.core.model.source_learning.event import (
    SourceEventFlashcardRequest,
    SourceEventLearningFact,
    SourceEventLearningFactOccurrence,
    SourceEventLearningFactRequest,
)
from kms2.core.model.source_learning.flashcard import (
    SourceFlashcard,
    SourceFlashcardOccurrence,
)
from kms2.database.source_learning.repository import SourceLearningRepository
from kms2.langgraph.source_learning.state import (
    SourceEventFlashcardWorkerResult,
    SourceEventLearningFactWorkerResult,
    SourceLearningState,
)
from kms2.module.source_learning.event import (
    SourceEventFlashcardModule,
    SourceEventLearningFactModule,
)


class EventLearningFactWorkerState(TypedDict):
    """Backend request sent to one event learning-fact worker."""

    ordinal: int
    request: SourceEventLearningFactRequest


class EventFlashcardWorkerState(TypedDict):
    """Backend request sent to one event card worker."""

    ordinal: int
    request: SourceEventFlashcardRequest


class SourceEventLearningFactNode:
    """Infer event learning facts and attach their originating provenance."""

    def __init__(
        self,
        repository: SourceLearningRepository,
        module: SourceEventLearningFactModule,
    ) -> None:
        self._repository = repository
        self._module = module

    async def load(self, state: SourceLearningState) -> dict:
        """Load distinct event hub/source-fact requests."""
        return {
            'event_learning_fact_requests': await self._repository.load_event_learning_requests(
                state.source_uuid
            )
        }

    def dispatch(
        self, state: SourceLearningState
    ) -> list[Send] | Literal['source_event_learning_fact_collect']:
        """Dispatch one worker per request in stable order."""
        return [
            Send(
                'source_event_learning_fact_worker',
                {'ordinal': ordinal, 'request': request},
            )
            for ordinal, request in enumerate(
                state.event_learning_fact_requests
            )
        ] or 'source_event_learning_fact_collect'

    async def worker(
        self, state: EventLearningFactWorkerState
    ) -> dict[str, list[SourceEventLearningFactWorkerResult]]:
        """Infer from content only, retaining backend request order."""
        facts = await self._module.aforward(
            request=state['request'].model_input()
        )
        return {
            'event_learning_fact_results': [
                SourceEventLearningFactWorkerResult(
                    ordinal=state['ordinal'], facts=facts
                )
            ]
        }

    def collect(self, state: SourceLearningState) -> dict:
        """Materialize candidates with their original hub and source fact."""
        occurrences: list[SourceEventLearningFactOccurrence] = []
        for item in sorted(
            state.event_learning_fact_results, key=lambda item: item.ordinal
        ):
            request = state.event_learning_fact_requests[item.ordinal]
            for candidate in item.facts:
                occurrences.append(
                    SourceEventLearningFactOccurrence(
                        learning_fact=SourceEventLearningFact(
                            text=candidate.text
                        ),
                        hub_uuid=request.hub_uuid,
                        source_fact_uuid=request.source_fact_uuid,
                    )
                )
        return {'event_learning_fact_occurrences': occurrences}

    async def persist(self, state: SourceLearningState) -> dict[str, int]:
        """Persist completed event learning-fact occurrences."""
        count = await self._repository.persist_event_learning_facts(
            state.source_uuid, state.event_learning_fact_occurrences
        )
        return {'event_learning_fact_count': count}


class SourceEventFlashcardNode:
    """Infer event cards and attach their persisted learning-fact identity."""

    def __init__(
        self,
        repository: SourceLearningRepository,
        module: SourceEventFlashcardModule,
    ) -> None:
        self._repository = repository
        self._module = module

    async def load(self, state: SourceLearningState) -> dict:
        """Load persisted event learning-fact requests."""
        return {
            'event_flashcard_requests': await self._repository.load_event_flashcard_requests(
                state.source_uuid
            )
        }

    def dispatch(
        self, state: SourceLearningState
    ) -> list[Send] | Literal['source_event_flashcard_collect']:
        """Dispatch one card worker per persisted learning fact."""
        return [
            Send(
                'source_event_flashcard_worker',
                {'ordinal': ordinal, 'request': request},
            )
            for ordinal, request in enumerate(state.event_flashcard_requests)
        ] or 'source_event_flashcard_collect'

    async def worker(
        self, state: EventFlashcardWorkerState
    ) -> dict[str, list[SourceEventFlashcardWorkerResult]]:
        """Infer one card from only the persisted learning-fact text."""
        result = await self._module.aforward(
            request=state['request'].model_input()
        )
        return {
            'event_flashcard_results': [
                SourceEventFlashcardWorkerResult(
                    ordinal=state['ordinal'], result=result
                )
            ]
        }

    def collect(self, state: SourceLearningState) -> dict:
        """Materialize cards with their original persisted learning fact."""
        occurrences: list[SourceFlashcardOccurrence] = []
        for item in sorted(
            state.event_flashcard_results, key=lambda item: item.ordinal
        ):
            request = state.event_flashcard_requests[item.ordinal]
            occurrences.append(
                SourceFlashcardOccurrence(
                    card=SourceFlashcard(
                        question=item.result.question, answer=item.result.answer
                    ),
                    learning_fact_uuid=request.learning_fact.uuid,
                )
            )
        return {'event_flashcard_occurrences': occurrences}

    async def persist(self, state: SourceLearningState) -> dict[str, int]:
        """Persist completed cards without replacing collection-owned identities."""
        count = await self._repository.persist_flashcards(
            state.source_uuid, state.event_flashcard_occurrences
        )
        return {'event_flashcard_count': count}
