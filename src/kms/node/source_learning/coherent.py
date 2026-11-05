"""Bounded source-material coherent flashcard generation."""

from collections import defaultdict
from collections.abc import Iterable
from typing import Literal, TypedDict

import dspy
from langgraph.types import Send
from pydantic import TypeAdapter

from kms.core.content import MultimodalContentBuilder
from kms.core.media_budget import ImagePayloadBudget
from kms.core.model.source_learning.coherent import (
    SourceAtomicFlashcardPacket,
    SourceCoherentFlashcardRequest,
)
from kms.core.model.source_learning.flashcard import (
    SourceFlashcard,
    SourceFlashcardOccurrence,
)
from kms.core.tokenization.budget import InputBudgetExceeded, TokenBudget
from kms.langgraph.source_learning.state import (
    SourceCoherentFlashcardWorkerResult,
    SourceLearningState,
)
from kms.module.source_learning.coherent import SourceCoherentFlashcardModule
from kms.service.content import load_content_images, pack_content_batches


class SourceCoherentFlashcardWorkerState(TypedDict):
    """Backend request sent to one coherent flashcard worker."""

    ordinal: int
    request: SourceCoherentFlashcardRequest


class SourceCoherentFlashcardNode:
    """Generate additional cards from retained source-material packets."""

    def __init__(
        self,
        module: SourceCoherentFlashcardModule,
        *,
        budget: TokenBudget,
        image_budget: ImagePayloadBudget,
    ) -> None:
        self._module = module
        self._budget = budget
        self._image_budget = image_budget

    async def prepare(self, state: SourceLearningState) -> dict[str, object]:
        """Pack eligible whole atomic packets by page into coherent requests."""
        packets_by_page: dict[int, list[SourceAtomicFlashcardPacket]] = (
            defaultdict(list)
        )
        for packet in state.atomic_packets:
            if packet.cards:
                packets_by_page[packet.request.source_page_index].append(packet)

        requests: list[SourceCoherentFlashcardRequest] = []
        for source_page_index, packets in packets_by_page.items():

            def project(
                items: list[SourceAtomicFlashcardPacket],
                builder: MultimodalContentBuilder,
                *,
                page_index: int = source_page_index,
            ) -> dict[str, object]:
                request = SourceCoherentFlashcardRequest(
                    source_page_index=page_index,
                    packets=items,
                )
                return {'request': request.model_input(builder)}

            batches = await pack_content_batches(
                [packets],
                project=project,
                budget=self._budget,
                image_budget=self._image_budget,
                is_eligible=lambda batch: (
                    sum(len(packet.cards) for packet in batch) >= 2
                ),
            )
            requests.extend(
                SourceCoherentFlashcardRequest(
                    source_page_index=source_page_index,
                    packets=batch,
                )
                for batch in batches
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
        """Run one coherent inference request with its local native images."""
        builder = MultimodalContentBuilder()
        request = state['request'].model_input(builder)
        payload = {'request': request, 'content_images': []}
        serialized = (
            TypeAdapter(dict[str, object])
            .dump_json(payload, serialize_as_any=True)
            .decode('utf-8')
        )
        if self._budget.counter.estimate(
            text=serialized, images=builder.images
        ) > (self._budget.token_limit):
            raise InputBudgetExceeded(
                'Source context exceeds the configured input budget'
            )
        images = await load_content_images(
            builder.images, budget=self._image_budget
        )
        cards = await self._module.acall(
            request=request,
            content_images=[dspy.Image(image) for image in images],
        )
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
                source_block_uuids = _unique_values(
                    block_uuid
                    for parent in selected
                    for block_uuid in parent.source_block_uuids
                )
                occurrences.append(
                    SourceFlashcardOccurrence(
                        card=SourceFlashcard(
                            question=candidate.question,
                            answer=candidate.answer,
                        ),
                        source_block_uuids=source_block_uuids,
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
