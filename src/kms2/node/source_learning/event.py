"""LangGraph nodes for event learning facts and cards."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.core.model.source_learning.event import (
    SourceEventFlashcardInput,
    SourceEventLearningFactInput,
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
    """Input sent to one event learning-fact worker."""

    ordinal: int
    request: SourceEventLearningFactInput


class EventFlashcardWorkerState(TypedDict):
    """Input sent to one event card worker."""

    ordinal: int
    request: SourceEventFlashcardInput


class SourceEventLearningFactNode:
    """Load, infer, order, and persist event learning facts."""

    def __init__(
        self,
        repository: SourceLearningRepository,
        module: SourceEventLearningFactModule,
    ) -> None:
        self._repository = repository
        self._module = module

    async def load(self, state: SourceLearningState) -> dict:
        """Load event hub evidence."""
        return {
            'event_learning_fact_inputs': await self._repository.load_event_learning_inputs(
                state.source_uuid
            )
        }

    def dispatch(
        self, state: SourceLearningState
    ) -> list[Send] | Literal['source_event_learning_fact_collect']:
        """Dispatch one worker per event hub in stable order."""
        return [
            Send(
                'source_event_learning_fact_worker',
                {'ordinal': ordinal, 'request': request},
            )
            for ordinal, request in enumerate(state.event_learning_fact_inputs)
        ] or ['source_event_learning_fact_collect'][0]

    async def worker(self, state: EventLearningFactWorkerState) -> dict:
        """Run one event learning-fact inference."""
        return {
            'event_learning_fact_results': [
                SourceEventLearningFactWorkerResult(
                    ordinal=state['ordinal'],
                    result=await self._module.aforward(
                        request=state['request']
                    ),
                )
            ]
        }

    def collect(self, state: SourceLearningState) -> dict:
        """Restore event model results to hub order."""
        return {
            'event_learning_fact_results_ordered': [
                item.result
                for item in sorted(
                    state.event_learning_fact_results,
                    key=lambda item: item.ordinal,
                )
            ]
        }

    async def persist(self, state: SourceLearningState) -> dict[str, int]:
        """Persist event learning facts and selected evidence."""
        return {
            'event_learning_fact_count': await self._repository.replace_event_learning_facts(
                state.source_uuid,
                state.event_learning_fact_inputs,
                state.event_learning_fact_results_ordered,
            )
        }


class SourceEventFlashcardNode:
    """Load, infer, order, and persist event flashcards."""

    def __init__(
        self,
        repository: SourceLearningRepository,
        module: SourceEventFlashcardModule,
    ) -> None:
        self._repository = repository
        self._module = module

    async def load(self, state: SourceLearningState) -> dict:
        """Load persisted event learning facts."""
        return {
            'event_flashcard_inputs': await self._repository.load_event_flashcard_inputs(
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
            for ordinal, request in enumerate(state.event_flashcard_inputs)
        ] or ['source_event_flashcard_collect'][0]

    async def worker(self, state: EventFlashcardWorkerState) -> dict:
        """Run one event card inference."""
        return {
            'event_flashcard_results': [
                SourceEventFlashcardWorkerResult(
                    ordinal=state['ordinal'],
                    result=await self._module.aforward(
                        request=state['request']
                    ),
                )
            ]
        }

    def collect(self, state: SourceLearningState) -> dict:
        """Restore event card results to learning-fact order."""
        return {
            'event_flashcard_results_ordered': [
                item.result
                for item in sorted(
                    state.event_flashcard_results, key=lambda item: item.ordinal
                )
            ]
        }

    async def persist(self, state: SourceLearningState) -> dict[str, int]:
        """Persist one event card for each learning fact."""
        return {
            'event_flashcard_count': await self._repository.persist_event_flashcards(
                state.source_uuid,
                state.event_flashcard_inputs,
                state.event_flashcard_results_ordered,
            )
        }
