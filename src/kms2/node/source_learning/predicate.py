"""LangGraph nodes for predicate learning facts and cards."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.core.model.source_learning.predicate import (
    SourcePredicateFlashcardInput,
    SourcePredicateLearningFactInput,
)
from kms2.database.source_learning.repository import SourceLearningRepository
from kms2.langgraph.source_learning.state import (
    SourceLearningState,
    SourcePredicateFlashcardWorkerResult,
    SourcePredicateLearningFactWorkerResult,
)
from kms2.module.source_learning.predicate import (
    SourcePredicateFlashcardModule,
    SourcePredicateLearningFactModule,
)


class PredicateLearningFactWorkerState(TypedDict):
    """Input sent to one predicate learning-fact worker."""

    ordinal: int
    request: SourcePredicateLearningFactInput


class PredicateFlashcardWorkerState(TypedDict):
    """Input sent to one predicate card worker."""

    ordinal: int
    request: SourcePredicateFlashcardInput


class SourcePredicateLearningFactNode:
    """Load, infer, order, and persist predicate learning facts."""

    def __init__(
        self,
        repository: SourceLearningRepository,
        module: SourcePredicateLearningFactModule,
    ) -> None:
        self._repository = repository
        self._module = module

    async def load(self, state: SourceLearningState) -> dict:
        """Load predicate hub evidence."""
        return {
            'predicate_learning_fact_inputs': await self._repository.load_predicate_learning_inputs(
                state.source_uuid
            )
        }

    def dispatch(
        self, state: SourceLearningState
    ) -> list[Send] | Literal['source_predicate_learning_fact_collect']:
        """Dispatch one worker per predicate hub in stable order."""
        return [
            Send(
                'source_predicate_learning_fact_worker',
                {'ordinal': ordinal, 'request': request},
            )
            for ordinal, request in enumerate(
                state.predicate_learning_fact_inputs
            )
        ] or ['source_predicate_learning_fact_collect'][0]

    async def worker(self, state: PredicateLearningFactWorkerState) -> dict:
        """Run one predicate learning-fact inference."""
        return {
            'predicate_learning_fact_results': [
                SourcePredicateLearningFactWorkerResult(
                    ordinal=state['ordinal'],
                    result=await self._module.aforward(
                        request=state['request']
                    ),
                )
            ]
        }

    def collect(self, state: SourceLearningState) -> dict:
        """Restore predicate model results to hub order."""
        return {
            'predicate_learning_fact_results_ordered': [
                item.result
                for item in sorted(
                    state.predicate_learning_fact_results,
                    key=lambda item: item.ordinal,
                )
            ]
        }

    async def persist(self, state: SourceLearningState) -> dict[str, int]:
        """Persist predicate learning facts and selected evidence."""
        return {
            'predicate_learning_fact_count': await self._repository.replace_predicate_learning_facts(
                state.source_uuid,
                state.predicate_learning_fact_inputs,
                state.predicate_learning_fact_results_ordered,
            )
        }


class SourcePredicateFlashcardNode:
    """Load, infer, order, and persist predicate flashcards."""

    def __init__(
        self,
        repository: SourceLearningRepository,
        module: SourcePredicateFlashcardModule,
    ) -> None:
        self._repository = repository
        self._module = module

    async def load(self, state: SourceLearningState) -> dict:
        """Load persisted predicate learning facts."""
        return {
            'predicate_flashcard_inputs': await self._repository.load_predicate_flashcard_inputs(
                state.source_uuid
            )
        }

    def dispatch(
        self, state: SourceLearningState
    ) -> list[Send] | Literal['source_predicate_flashcard_collect']:
        """Dispatch one card worker per persisted learning fact."""
        return [
            Send(
                'source_predicate_flashcard_worker',
                {'ordinal': ordinal, 'request': request},
            )
            for ordinal, request in enumerate(state.predicate_flashcard_inputs)
        ] or ['source_predicate_flashcard_collect'][0]

    async def worker(self, state: PredicateFlashcardWorkerState) -> dict:
        """Run one predicate card inference."""
        return {
            'predicate_flashcard_results': [
                SourcePredicateFlashcardWorkerResult(
                    ordinal=state['ordinal'],
                    result=await self._module.aforward(
                        request=state['request']
                    ),
                )
            ]
        }

    def collect(self, state: SourceLearningState) -> dict:
        """Restore predicate card results to learning-fact order."""
        return {
            'predicate_flashcard_results_ordered': [
                item.result
                for item in sorted(
                    state.predicate_flashcard_results,
                    key=lambda item: item.ordinal,
                )
            ]
        }

    async def persist(self, state: SourceLearningState) -> dict[str, int]:
        """Persist one predicate card for each learning fact."""
        return {
            'predicate_flashcard_count': await self._repository.persist_predicate_flashcards(
                state.source_uuid,
                state.predicate_flashcard_inputs,
                state.predicate_flashcard_results_ordered,
            )
        }
