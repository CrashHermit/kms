"""Bounded atomic flashcard generation node."""

from typing import Literal, TypedDict

import dspy
from langgraph.types import Send
from pydantic import TypeAdapter

from kms.config.inference import ContextWindowSettings
from kms.core.content import MultimodalContentBuilder
from kms.core.media_budget import ImagePayloadBudget
from kms.core.model.source_learning.atomic import SourceAtomicFlashcardRequest
from kms.core.model.source_learning.coherent import SourceAtomicFlashcardPacket
from kms.core.model.source_learning.flashcard import (
    SourceFlashcard,
    SourceFlashcardOccurrence,
)
from kms.core.source_context import count_source_tokens
from kms.core.tokenization.budget import InputBudgetExceeded, TokenBudget
from kms.core.windowing import select_window
from kms.database.source.source_block_repository import SourceBlockRepository
from kms.database.source_learning.repository import SourceLearningRepository
from kms.langgraph.source_learning.state import (
    SourceAtomicFlashcardWorkerResult,
    SourceLearningState,
)
from kms.module.source_learning.atomic import SourceAtomicFlashcardModule
from kms.service.content import load_content_images, pack_content_batches


class SourceAtomicFlashcardWorkerState(TypedDict):
    """Backend request sent to one atomic flashcard worker."""

    ordinal: int
    request: SourceAtomicFlashcardRequest


class SourceAtomicFlashcardNode:
    """Generate ordered atomic cards from source-material requests."""

    def __init__(
        self,
        repository: SourceLearningRepository,
        module: SourceAtomicFlashcardModule,
        context_window: ContextWindowSettings,
        source_block_repository: SourceBlockRepository,
        *,
        budget: TokenBudget,
        image_budget: ImagePayloadBudget,
    ) -> None:
        self._repository = repository
        self._module = module
        self._context_window = context_window
        self._source_block_repository = source_block_repository
        self._budget = budget
        self._image_budget = image_budget

    async def load(self, state: SourceLearningState) -> dict[str, object]:
        """Load metadata and retain canonical source context windows."""
        materials = await self._repository.load_atomic_flashcard_materials(
            state.source_uuid
        )
        if not materials:
            return {'atomic_requests': []}
        blocks = await self._source_block_repository.load_blocks(
            state.source_uuid
        )
        token_counts = count_source_tokens(blocks, (self._budget.counter,))
        positions = {block.uuid: index for index, block in enumerate(blocks)}
        requests: list[SourceAtomicFlashcardRequest] = []
        for material in materials:
            position = positions[material.source_block_uuid]
            window = select_window(
                blocks,
                [position],
                token_counts=token_counts,
                backward_budget=self._context_window.backward_budget,
                forward_budget=self._context_window.forward_budget,
                target_budget=0,
            )
            requests.append(
                SourceAtomicFlashcardRequest(
                    **material.model_dump(),
                    window=window,
                )
            )
        return {'atomic_requests': requests}

    async def dispatch(
        self, state: SourceLearningState
    ) -> list[Send] | Literal['source_atomic_flashcard_collect']:
        """Refine one-request atomic batches by complete text/image payloads."""
        requests = state.atomic_requests
        if not requests:
            return 'source_atomic_flashcard_collect'

        def project(
            items: list[SourceAtomicFlashcardRequest],
            builder: MultimodalContentBuilder,
        ) -> dict[str, object]:
            return {'request': items[0].model_input(builder)}

        batches = await pack_content_batches(
            [[request] for request in requests],
            project=project,
            budget=self._budget,
            image_budget=self._image_budget,
        )
        sends = [
            Send(
                'source_atomic_flashcard_worker',
                {'ordinal': ordinal, 'request': batch[0]},
            )
            for ordinal, batch in enumerate(batches)
        ]
        return sends or 'source_atomic_flashcard_collect'

    async def worker(
        self, state: SourceAtomicFlashcardWorkerState
    ) -> dict[str, list[SourceAtomicFlashcardWorkerResult]]:
        """Run one atomic inference request with its local native images."""
        builder = MultimodalContentBuilder()
        request = state['request'].model_input(builder)
        payload = {
            'request': request,
            'content_images': [],
        }
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
                    source_block_uuids=[request.source_block_uuid],
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
