"""LangGraph nodes for predicate learning facts and cards."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.core.model.source_learning.flashcard import (
    SourceFlashcard,
    SourceFlashcardOccurrence,
)
from kms2.core.model.source_learning.predicate import (
    SourcePredicateFlashcardRequest,
    SourcePredicateLearningFact,
    SourcePredicateLearningFactOccurrence,
    SourcePredicateLearningFactRequest,
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
    """Backend request sent to one predicate learning-fact worker."""

    ordinal: int
    request: SourcePredicateLearningFactRequest


class PredicateFlashcardWorkerState(TypedDict):
    """Backend request sent to one predicate card worker."""

    ordinal: int
    request: SourcePredicateFlashcardRequest


class SourcePredicateLearningFactNode:
    """Infer predicate learning facts and attach their originating provenance."""

    def __init__(
        self,
        repository: SourceLearningRepository,
        module: SourcePredicateLearningFactModule,
    ) -> None:
        self._repository = repository
        self._module = module

    async def load(self, state: SourceLearningState) -> dict:
        """Load distinct predicate hub/source-fact requests."""
        return {
            'predicate_learning_fact_requests': await self._repository.load_predicate_learning_requests(
                state.source_uuid
            )
        }

    def dispatch(
        self, state: SourceLearningState
    ) -> list[Send] | Literal['source_predicate_learning_fact_collect']:
        """Dispatch one worker per request in stable order."""
        return [
            Send(
                'source_predicate_learning_fact_worker',
                {'ordinal': ordinal, 'request': request},
            )
            for ordinal, request in enumerate(
                state.predicate_learning_fact_requests
            )
        ] or 'source_predicate_learning_fact_collect'

    async def worker(
        self, state: PredicateLearningFactWorkerState
    ) -> dict[str, list[SourcePredicateLearningFactWorkerResult]]:
        """Infer from content only, retaining backend request order."""
        facts = await self._module.aforward(
            request=state['request'].model_input()
        )
        return {
            'predicate_learning_fact_results': [
                SourcePredicateLearningFactWorkerResult(
                    ordinal=state['ordinal'], facts=facts
                )
            ]
        }

    def collect(self, state: SourceLearningState) -> dict:
        """Materialize candidates with their original hub and source fact."""
        occurrences: list[SourcePredicateLearningFactOccurrence] = []
        for item in sorted(
            state.predicate_learning_fact_results, key=lambda item: item.ordinal
        ):
            request = state.predicate_learning_fact_requests[item.ordinal]
            for candidate in item.facts:
                occurrences.append(
                    SourcePredicateLearningFactOccurrence(
                        learning_fact=SourcePredicateLearningFact(
                            text=candidate.text
                        ),
                        hub_uuid=request.hub_uuid,
                        source_fact_uuid=request.source_fact_uuid,
                    )
                )
        return {'predicate_learning_fact_occurrences': occurrences}

    async def persist(self, state: SourceLearningState) -> dict[str, int]:
        """Persist completed predicate learning-fact occurrences."""
        count = await self._repository.persist_predicate_learning_facts(
            state.source_uuid, state.predicate_learning_fact_occurrences
        )
        return {'predicate_learning_fact_count': count}


class SourcePredicateFlashcardNode:
    """Infer predicate cards and attach their persisted learning-fact identity."""

    def __init__(
        self,
        repository: SourceLearningRepository,
        module: SourcePredicateFlashcardModule,
    ) -> None:
        self._repository = repository
        self._module = module

    async def load(self, state: SourceLearningState) -> dict:
        """Load persisted predicate learning-fact requests."""
        return {
            'predicate_flashcard_requests': await self._repository.load_predicate_flashcard_requests(
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
            for ordinal, request in enumerate(
                state.predicate_flashcard_requests
            )
        ] or 'source_predicate_flashcard_collect'

    async def worker(
        self, state: PredicateFlashcardWorkerState
    ) -> dict[str, list[SourcePredicateFlashcardWorkerResult]]:
        """Infer one card from only the persisted learning-fact text."""
        result = await self._module.aforward(
            request=state['request'].model_input()
        )
        return {
            'predicate_flashcard_results': [
                SourcePredicateFlashcardWorkerResult(
                    ordinal=state['ordinal'], result=result
                )
            ]
        }

    def collect(self, state: SourceLearningState) -> dict:
        """Materialize cards with their original persisted learning fact."""
        occurrences: list[SourceFlashcardOccurrence] = []
        for item in sorted(
            state.predicate_flashcard_results, key=lambda item: item.ordinal
        ):
            request = state.predicate_flashcard_requests[item.ordinal]
            occurrences.append(
                SourceFlashcardOccurrence(
                    card=SourceFlashcard(
                        question=item.result.question, answer=item.result.answer
                    ),
                    learning_fact_uuid=request.learning_fact.uuid,
                )
            )
        return {'predicate_flashcard_occurrences': occurrences}

    async def persist(self, state: SourceLearningState) -> dict[str, int]:
        """Persist completed cards without replacing collection-owned identities."""
        count = await self._repository.persist_flashcards(
            state.source_uuid, state.predicate_flashcard_occurrences
        )
        return {'predicate_flashcard_count': count}
