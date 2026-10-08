"""Bounded same-hub coherent flashcard generation."""

from collections import defaultdict
from collections.abc import Iterable
from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.core.model.source_learning.coherent import (
    SourceAtomicFlashcardPacket,
    SourceCoherentFlashcardRequest,
)
from kms2.core.model.source_learning.flashcard import (
    SourceFlashcard,
    SourceFlashcardOccurrence,
)
from kms2.core.windowing import TokenBudget, pack_items
from kms2.langgraph.source_learning.state import (
    SourceCoherentFlashcardWorkerResult,
    SourceLearningState,
)
from kms2.module.source_learning.coherent import SourceCoherentFlashcardModule


class SourceCoherentFlashcardWorkerState(TypedDict):
    """Backend request sent to one coherent flashcard worker."""

    ordinal: int
    request: SourceCoherentFlashcardRequest


class SourceCoherentFlashcardNode:
    """Generate additional cards from retained atomic packets within each hub."""

    def __init__(
        self,
        module: SourceCoherentFlashcardModule,
        *,
        budget: TokenBudget,
    ) -> None:
        self._module = module
        self._budget = budget

    def prepare(self, state: SourceLearningState) -> dict[str, object]:
        """Pack whole atomic packets into same-hub coherent requests."""
        packets_by_hub: dict[str, list[SourceAtomicFlashcardPacket]] = (
            defaultdict(list)
        )
        for packet in state.atomic_packets:
            if packet.cards:
                packets_by_hub[packet.request.hub_uuid].append(packet)

        requests: list[SourceCoherentFlashcardRequest] = []
        for hub_uuid, packets in packets_by_hub.items():
            context = packets[0].request.input.context
            whole_request = SourceCoherentFlashcardRequest(
                hub_uuid=hub_uuid,
                context=context,
                packets=packets,
            )
            whole_input = whole_request.model_input()
            costs = self._budget.counter.count_texts(
                [
                    whole_input.context.model_dump_json(),
                    *[
                        packet.model_dump_json(
                            exclude={'cards': {'__all__': {'index'}}}
                        )
                        for packet in whole_input.packets
                    ],
                ]
            )
            context_cost, *packet_costs = costs
            batches = pack_items(
                packets,
                token_counts=packet_costs,
                token_budget=self._budget.token_limit - context_cost,
            )
            for batch in batches:
                if sum(len(packet.cards) for packet in batch) < 2:
                    continue
                requests.append(
                    SourceCoherentFlashcardRequest(
                        hub_uuid=hub_uuid,
                        context=context,
                        packets=batch,
                    )
                )
        return {'coherent_requests': requests}

    def dispatch(
        self, state: SourceLearningState
    ) -> list[Send] | Literal['source_coherent_flashcard_collect']:
        """Dispatch one stable worker per prepared coherent request."""
        return [
            Send(
                'source_coherent_flashcard_worker',
                {'ordinal': ordinal, 'request': request},
            )
            for ordinal, request in enumerate(state.coherent_requests)
        ] or 'source_coherent_flashcard_collect'

    async def worker(
        self, state: SourceCoherentFlashcardWorkerState
    ) -> dict[str, list[SourceCoherentFlashcardWorkerResult]]:
        """Run one content-only coherent inference request."""
        cards = await self._module.acall(request=state['request'].model_input())
        return {
            'coherent_results': [
                SourceCoherentFlashcardWorkerResult(
                    ordinal=state['ordinal'],
                    cards=cards,
                )
            ]
        }

    def collect(self, state: SourceLearningState) -> dict[str, object]:
        """Resolve local parent indexes and attach exact selected provenance."""
        occurrences: list[SourceFlashcardOccurrence] = []
        for result in sorted(
            state.coherent_results,
            key=lambda item: item.ordinal,
        ):
            request = state.coherent_requests[result.ordinal]
            parents = [
                occurrence
                for packet in request.packets
                for occurrence in packet.cards
            ]
            for candidate in result.cards:
                selected = [
                    parents[index - 1] for index in candidate.card_indexes
                ]
                triplet_uuids = _unique_values(
                    triplet_uuid
                    for parent in selected
                    for triplet_uuid in parent.triplet_uuids
                )
                source_fact_uuids = _unique_values(
                    source_fact_uuid
                    for parent in selected
                    for source_fact_uuid in parent.source_fact_uuids
                )
                occurrences.append(
                    SourceFlashcardOccurrence(
                        card=SourceFlashcard(
                            question=candidate.question,
                            answer=candidate.answer,
                        ),
                        hub_uuid=request.hub_uuid,
                        triplet_uuids=triplet_uuids,
                        source_fact_uuids=source_fact_uuids,
                        derived_card_uuids=_unique_values(
                            parent.card.uuid for parent in selected
                        ),
                    )
                )
        return {'coherent_occurrences': occurrences}


def _unique_values(values: Iterable[str]) -> list[str]:
    """Return first-use-ordered distinct nonempty values."""
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique
