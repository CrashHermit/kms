"""LangGraph nodes for triplet learning facts and cards."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.core.model.source_learning.flashcard import (
    SourceFlashcard,
    SourceFlashcardOccurrence,
)
from kms2.core.model.source_learning.triplet import (
    SourceTripletFlashcardRequest,
    SourceTripletLearningFact,
    SourceTripletLearningFactOccurrence,
    SourceTripletLearningFactRequest,
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
    """Backend request sent to one triplet learning-fact worker."""

    ordinal: int
    request: SourceTripletLearningFactRequest


class TripletFlashcardWorkerState(TypedDict):
    """Backend request sent to one triplet card worker."""

    ordinal: int
    request: SourceTripletFlashcardRequest


class SourceTripletLearningFactNode:
    """Infer triplet learning facts and attach their originating provenance."""

    def __init__(
        self,
        repository: SourceLearningRepository,
        module: SourceTripletLearningFactModule,
    ) -> None:
        self._repository = repository
        self._module = module

    async def load(self, state: SourceLearningState) -> dict:
        """Load distinct triplet hub/source-fact requests."""
        return {
            'triplet_learning_fact_requests': await self._repository.load_triplet_learning_requests(
                state.source_uuid
            )
        }

    def dispatch(
        self, state: SourceLearningState
    ) -> list[Send] | Literal['source_triplet_learning_fact_collect']:
        """Dispatch one worker per request in stable order."""
        return [
            Send(
                'source_triplet_learning_fact_worker',
                {'ordinal': ordinal, 'request': request},
            )
            for ordinal, request in enumerate(
                state.triplet_learning_fact_requests
            )
        ] or 'source_triplet_learning_fact_collect'

    async def worker(
        self, state: TripletLearningFactWorkerState
    ) -> dict[str, list[SourceTripletLearningFactWorkerResult]]:
        """Infer from content only, retaining backend request order."""
        facts = await self._module.aforward(
            request=state['request'].model_input()
        )
        return {
            'triplet_learning_fact_results': [
                SourceTripletLearningFactWorkerResult(
                    ordinal=state['ordinal'], facts=facts
                )
            ]
        }

    def collect(self, state: SourceLearningState) -> dict:
        """Materialize candidates with their original hub and source fact."""
        occurrences: list[SourceTripletLearningFactOccurrence] = []
        for item in sorted(
            state.triplet_learning_fact_results, key=lambda item: item.ordinal
        ):
            request = state.triplet_learning_fact_requests[item.ordinal]
            for candidate in item.facts:
                occurrences.append(
                    SourceTripletLearningFactOccurrence(
                        learning_fact=SourceTripletLearningFact(
                            text=candidate.text
                        ),
                        hub_uuid=request.hub_uuid,
                        source_fact_uuid=request.source_fact_uuid,
                    )
                )
        return {'triplet_learning_fact_occurrences': occurrences}

    async def persist(self, state: SourceLearningState) -> dict[str, int]:
        """Persist completed triplet learning-fact occurrences."""
        count = await self._repository.persist_triplet_learning_facts(
            state.source_uuid, state.triplet_learning_fact_occurrences
        )
        return {'triplet_learning_fact_count': count}


class SourceTripletFlashcardNode:
    """Infer triplet cards and attach their persisted learning-fact identity."""

    def __init__(
        self,
        repository: SourceLearningRepository,
        module: SourceTripletFlashcardModule,
    ) -> None:
        self._repository = repository
        self._module = module

    async def load(self, state: SourceLearningState) -> dict:
        """Load persisted triplet learning-fact requests."""
        return {
            'triplet_flashcard_requests': await self._repository.load_triplet_flashcard_requests(
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
            for ordinal, request in enumerate(state.triplet_flashcard_requests)
        ] or 'source_triplet_flashcard_collect'

    async def worker(
        self, state: TripletFlashcardWorkerState
    ) -> dict[str, list[SourceTripletFlashcardWorkerResult]]:
        """Infer one card from only the persisted learning-fact text."""
        result = await self._module.aforward(
            request=state['request'].model_input()
        )
        return {
            'triplet_flashcard_results': [
                SourceTripletFlashcardWorkerResult(
                    ordinal=state['ordinal'], result=result
                )
            ]
        }

    def collect(self, state: SourceLearningState) -> dict:
        """Materialize cards with their original persisted learning fact."""
        occurrences: list[SourceFlashcardOccurrence] = []
        for item in sorted(
            state.triplet_flashcard_results, key=lambda item: item.ordinal
        ):
            request = state.triplet_flashcard_requests[item.ordinal]
            occurrences.append(
                SourceFlashcardOccurrence(
                    card=SourceFlashcard(
                        question=item.result.question, answer=item.result.answer
                    ),
                    learning_fact_uuid=request.learning_fact.uuid,
                )
            )
        return {'triplet_flashcard_occurrences': occurrences}

    async def persist(self, state: SourceLearningState) -> dict[str, int]:
        """Persist completed cards without replacing collection-owned identities."""
        count = await self._repository.persist_flashcards(
            state.source_uuid, state.triplet_flashcard_occurrences
        )
        return {'triplet_flashcard_count': count}
