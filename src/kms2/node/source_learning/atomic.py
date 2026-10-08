"""Bounded atomic flashcard generation node."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.core.model.source_learning.atomic import SourceAtomicFlashcardRequest
from kms2.core.model.source_learning.coherent import SourceAtomicFlashcardPacket
from kms2.core.model.source_learning.flashcard import (
    SourceFlashcard,
    SourceFlashcardOccurrence,
)
from kms2.core.windowing import TokenBudget, pack_items
from kms2.database.source_learning.repository import SourceLearningRepository
from kms2.langgraph.source_learning.state import (
    SourceAtomicFlashcardWorkerResult,
    SourceLearningState,
)
from kms2.module.source_learning.atomic import SourceAtomicFlashcardModule


class SourceAtomicFlashcardWorkerState(TypedDict):
    """Backend request sent to one atomic flashcard worker."""

    ordinal: int
    request: SourceAtomicFlashcardRequest


class SourceAtomicFlashcardNode:
    """Generate ordered atomic cards from exact source-semantic requests."""

    def __init__(
        self,
        repository: SourceLearningRepository,
        module: SourceAtomicFlashcardModule,
        *,
        budget: TokenBudget,
    ) -> None:
        self._repository = repository
        self._module = module
        self._budget = budget

    async def load(self, state: SourceLearningState) -> dict[str, object]:
        """Load ordered atomic requests for the source."""
        return {
            'atomic_requests': await self._repository.load_atomic_flashcard_requests(
                state.source_uuid
            )
        }

    def dispatch(
        self, state: SourceLearningState
    ) -> list[Send] | Literal['source_atomic_flashcard_collect']:
        """Pack complete atomic inputs and dispatch stable worker ordinals."""
        requests = state.atomic_requests
        if not requests:
            return 'source_atomic_flashcard_collect'
        costs = self._budget.counter.count_texts(
            [request.model_input().model_dump_json() for request in requests]
        )
        batches = pack_items(
            requests,
            token_counts=costs,
            token_budget=self._budget.token_limit,
            max_items=1,
        )
        sends: list[Send] = []
        ordinal = 0
        for batch in batches:
            for request in batch:
                sends.append(
                    Send(
                        'source_atomic_flashcard_worker',
                        {'ordinal': ordinal, 'request': request},
                    )
                )
                ordinal += 1
        return sends or 'source_atomic_flashcard_collect'

    async def worker(
        self, state: SourceAtomicFlashcardWorkerState
    ) -> dict[str, list[SourceAtomicFlashcardWorkerResult]]:
        """Run one content-only atomic inference request."""
        cards = await self._module.acall(request=state['request'].model_input())
        return {
            'atomic_results': [
                SourceAtomicFlashcardWorkerResult(
                    ordinal=state['ordinal'],
                    cards=cards,
                )
            ]
        }

    def collect(self, state: SourceLearningState) -> dict[str, object]:
        """Materialize generic cards and retain every atomic packet."""
        packets: list[SourceAtomicFlashcardPacket] = []
        for result in sorted(
            state.atomic_results, key=lambda item: item.ordinal
        ):
            request = state.atomic_requests[result.ordinal]
            occurrences = [
                SourceFlashcardOccurrence(
                    card=SourceFlashcard(
                        question=candidate.question,
                        answer=candidate.answer,
                    ),
                    hub_uuid=request.hub_uuid,
                    triplet_uuids=[request.triplet_uuid],
                    source_fact_uuids=[request.source_fact_uuid],
                )
                for candidate in result.cards
            ]
            packets.append(
                SourceAtomicFlashcardPacket(
                    request=request,
                    cards=occurrences,
                )
            )
        return {'atomic_packets': packets}
