"""LangGraph nodes for triplet learning facts and cards."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.core.model.source_learning.triplet import (
    SourceTripletFlashcardInput,
    SourceTripletLearningFactInput,
)
from kms2.database.source_learning.repository import SourceLearningRepository
from kms2.langgraph.source_learning.state import (
    SourceLearningState,
    SourceTripletFlashcardWorkerResult,
    SourceTripletLearningFactWorkerResult,
)
from kms2.module.source_learning.triplet import (
    SourceTripletFlashcardModule,
    SourceTripletLearningFactModule,
)


class TripletLearningFactWorkerState(TypedDict):
    """Input sent to one triplet learning-fact worker."""

    ordinal: int
    request: SourceTripletLearningFactInput


class TripletFlashcardWorkerState(TypedDict):
    """Input sent to one triplet card worker."""

    ordinal: int
    request: SourceTripletFlashcardInput


class SourceTripletLearningFactNode:
    """Load, infer, order, and persist triplet learning facts."""

    def __init__(
        self,
        repository: SourceLearningRepository,
        module: SourceTripletLearningFactModule,
    ) -> None:
        self._repository = repository
        self._module = module

    async def load(self, state: SourceLearningState) -> dict:
        """Load triplet hub evidence."""
        return {
            'triplet_learning_fact_inputs': await self._repository.load_triplet_learning_inputs(
                state.source_uuid
            )
        }

    def dispatch(
        self, state: SourceLearningState
    ) -> list[Send] | Literal['source_triplet_learning_fact_collect']:
        """Dispatch one worker per triplet hub in stable order."""
        return [
            Send(
                'source_triplet_learning_fact_worker',
                {'ordinal': ordinal, 'request': request},
            )
            for ordinal, request in enumerate(
                state.triplet_learning_fact_inputs
            )
        ] or ['source_triplet_learning_fact_collect'][0]

    async def worker(self, state: TripletLearningFactWorkerState) -> dict:
        """Run one triplet learning-fact inference."""
        return {
            'triplet_learning_fact_results': [
                SourceTripletLearningFactWorkerResult(
                    ordinal=state['ordinal'],
                    result=await self._module.aforward(
                        request=state['request']
                    ),
                )
            ]
        }

    def collect(self, state: SourceLearningState) -> dict:
        """Restore triplet model results to hub order."""
        return {
            'triplet_learning_fact_results_ordered': [
                item.result
                for item in sorted(
                    state.triplet_learning_fact_results,
                    key=lambda item: item.ordinal,
                )
            ]
        }

    async def persist(self, state: SourceLearningState) -> dict[str, int]:
        """Persist triplet learning facts and selected evidence."""
        return {
            'triplet_learning_fact_count': await self._repository.replace_triplet_learning_facts(
                state.source_uuid,
                state.triplet_learning_fact_inputs,
                state.triplet_learning_fact_results_ordered,
            )
        }


class SourceTripletFlashcardNode:
    """Load, infer, order, and persist triplet flashcards."""

    def __init__(
        self,
        repository: SourceLearningRepository,
        module: SourceTripletFlashcardModule,
    ) -> None:
        self._repository = repository
        self._module = module

    async def load(self, state: SourceLearningState) -> dict:
        """Load persisted triplet learning facts."""
        return {
            'triplet_flashcard_inputs': await self._repository.load_triplet_flashcard_inputs(
                state.source_uuid
            )
        }

    def dispatch(
        self, state: SourceLearningState
    ) -> list[Send] | Literal['source_triplet_flashcard_collect']:
        """Dispatch one card worker per persisted learning fact."""
        return [
            Send(
                'source_triplet_flashcard_worker',
                {'ordinal': ordinal, 'request': request},
            )
            for ordinal, request in enumerate(state.triplet_flashcard_inputs)
        ] or ['source_triplet_flashcard_collect'][0]

    async def worker(self, state: TripletFlashcardWorkerState) -> dict:
        """Run one triplet card inference."""
        return {
            'triplet_flashcard_results': [
                SourceTripletFlashcardWorkerResult(
                    ordinal=state['ordinal'],
                    result=await self._module.aforward(
                        request=state['request']
                    ),
                )
            ]
        }

    def collect(self, state: SourceLearningState) -> dict:
        """Restore triplet card results to learning-fact order."""
        return {
            'triplet_flashcard_results_ordered': [
                item.result
                for item in sorted(
                    state.triplet_flashcard_results,
                    key=lambda item: item.ordinal,
                )
            ]
        }

    async def persist(self, state: SourceLearningState) -> dict[str, int]:
        """Persist one triplet card for each learning fact."""
        return {
            'triplet_flashcard_count': await self._repository.persist_triplet_flashcards(
                state.source_uuid,
                state.triplet_flashcard_inputs,
                state.triplet_flashcard_results_ordered,
            )
        }
