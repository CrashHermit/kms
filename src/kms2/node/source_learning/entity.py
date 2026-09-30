"""LangGraph nodes for entity learning facts and cards."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.core.model.source_learning.entity import (
    SourceEntityFlashcardInput,
    SourceEntityLearningFactInput,
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
    """Input sent to one entity learning-fact worker."""

    ordinal: int
    request: SourceEntityLearningFactInput


class EntityFlashcardWorkerState(TypedDict):
    """Input sent to one entity card worker."""

    ordinal: int
    request: SourceEntityFlashcardInput


class SourceEntityLearningFactNode:
    """Load, infer, order, and persist entity learning facts."""

    def __init__(
        self,
        repository: SourceLearningRepository,
        module: SourceEntityLearningFactModule,
    ) -> None:
        self._repository = repository
        self._module = module

    async def load(self, state: SourceLearningState) -> dict:
        """Load entity hub evidence."""
        return {
            'entity_learning_fact_inputs': await self._repository.load_entity_learning_inputs(
                state.source_uuid
            )
        }

    def dispatch(
        self, state: SourceLearningState
    ) -> list[Send] | Literal['source_entity_learning_fact_collect']:
        """Dispatch one worker per entity hub in stable order."""
        return [
            Send(
                'source_entity_learning_fact_worker',
                {'ordinal': ordinal, 'request': request},
            )
            for ordinal, request in enumerate(state.entity_learning_fact_inputs)
        ] or ['source_entity_learning_fact_collect'][0]

    async def worker(
        self, state: EntityLearningFactWorkerState
    ) -> dict[str, list[SourceEntityLearningFactWorkerResult]]:
        """Run one entity learning-fact inference."""
        result = await self._module.aforward(request=state['request'])
        return {
            'entity_learning_fact_results': [
                SourceEntityLearningFactWorkerResult(
                    ordinal=state['ordinal'], result=result
                )
            ]
        }

    def collect(self, state: SourceLearningState) -> dict:
        """Restore entity model results to hub order."""
        ordered = sorted(
            state.entity_learning_fact_results, key=lambda item: item.ordinal
        )
        return {
            'entity_learning_fact_results_ordered': [
                item.result for item in ordered
            ]
        }

    async def persist(self, state: SourceLearningState) -> dict[str, int]:
        """Persist entity learning facts and selected evidence."""
        count = await self._repository.persist_entity_learning_facts(
            state.source_uuid,
            state.entity_learning_fact_inputs,
            state.entity_learning_fact_results_ordered,
        )
        return {'entity_learning_fact_count': count}


class SourceEntityFlashcardNode:
    """Load, infer, order, and persist entity flashcards."""

    def __init__(
        self,
        repository: SourceLearningRepository,
        module: SourceEntityFlashcardModule,
    ) -> None:
        self._repository = repository
        self._module = module

    async def load(self, state: SourceLearningState) -> dict:
        """Load persisted entity learning facts."""
        return {
            'entity_flashcard_inputs': await self._repository.load_entity_flashcard_inputs(
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
            for ordinal, request in enumerate(state.entity_flashcard_inputs)
        ] or ['source_entity_flashcard_collect'][0]

    async def worker(
        self, state: EntityFlashcardWorkerState
    ) -> dict[str, list[SourceEntityFlashcardWorkerResult]]:
        """Run one entity card inference."""
        result = await self._module.aforward(request=state['request'])
        return {
            'entity_flashcard_results': [
                SourceEntityFlashcardWorkerResult(
                    ordinal=state['ordinal'], result=result
                )
            ]
        }

    def collect(self, state: SourceLearningState) -> dict:
        """Restore entity card results to learning-fact order."""
        ordered = sorted(
            state.entity_flashcard_results, key=lambda item: item.ordinal
        )
        return {
            'entity_flashcard_results_ordered': [
                item.result for item in ordered
            ]
        }

    async def persist(self, state: SourceLearningState) -> dict[str, int]:
        """Persist one entity card for each learning fact."""
        count = await self._repository.persist_entity_flashcards(
            state.source_uuid,
            state.entity_flashcard_inputs,
            state.entity_flashcard_results_ordered,
        )
        return {'entity_flashcard_count': count}
