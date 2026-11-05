"""LangGraph node for source-image descriptions."""

from typing import Literal, TypedDict

import dspy
from langgraph.types import Send
from pydantic import TypeAdapter

from kms.config.inference import ContextWindowSettings
from kms.core.content import MultimodalContentBuilder
from kms.core.media_budget import ImagePayloadBudget
from kms.core.model.block import SourceBlock
from kms.core.model.block_types import BlockType
from kms.core.model.page import SourcePage
from kms.core.model.source_processing.image_description import (
    ImageDescriptionInput,
    ImageDescriptionRequest,
    ImageDescriptionResult,
)
from kms.core.source_context import count_source_tokens, project_source_window
from kms.core.tokenization.budget import InputBudgetExceeded, TokenBudget
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.core.windowing import (
    pack_items,
    select_window,
)
from kms.langgraph.source_processing.state import SourceProcessingState
from kms.module.source_processing.image_description import (
    ImageDescriptionModule,
)
from kms.service.content import load_content_images, pack_content_batches


class ImageDescriptionWorkerState(TypedDict):
    """State supplied to one dispatched image description worker."""

    image_description_requests: list[ImageDescriptionRequest]


class ImageDescriptionNode:
    """Dispatch and collect source-image descriptions."""

    def __init__(
        self,
        describer: ImageDescriptionModule,
        context_window: ContextWindowSettings,
        *,
        budget: TokenBudget,
        image_budget: ImagePayloadBudget,
        batch_size: int,
        token_counters: tuple[MaterialTokenEstimator, ...],
    ) -> None:
        self._describer = describer
        self._context_window = context_window
        self._budget = budget
        self._image_budget = image_budget
        self._batch_size = batch_size
        self._token_counters = token_counters

    async def dispatch(
        self, state: SourceProcessingState
    ) -> list[Send] | Literal['image_description_collect']:
        """Dispatch image-description requests within text and image limits."""
        flat_blocks = [
            block
            for page in state.image_filtered_pages
            for block in page.blocks
        ]
        image_positions = [
            position
            for position, block in enumerate(flat_blocks)
            if block.block_type is BlockType.IMAGE and block.assets
        ]
        if not image_positions:
            return 'image_description_collect'
        token_counts = count_source_tokens(flat_blocks, self._token_counters)
        requests = [
            ImageDescriptionRequest(
                flat_position=position,
                source_block=flat_blocks[position],
                window=select_window(
                    flat_blocks,
                    [position],
                    token_counts=token_counts,
                    backward_budget=self._context_window.backward_budget,
                    forward_budget=self._context_window.forward_budget,
                ),
            )
            for position in image_positions
        ]

        def project(
            batch: list[ImageDescriptionRequest],
            builder: MultimodalContentBuilder,
        ) -> dict[str, object]:
            items = []
            for position, request in enumerate(batch):
                window = project_source_window(request.window, builder)
                items.append(
                    ImageDescriptionInput(
                        position=position,
                        target=window.target[0].material,
                        context_before=window.context_before,
                        context_after=window.context_after,
                    )
                )
            return {'items': items}

        # Keep the established item/text-window grouping; the source-context
        # packer then measures the full projected request and native assets.
        coarse_batches = pack_items(
            requests,
            token_counts=[1] * len(requests),
            token_budget=max(self._budget.token_limit, 1),
            max_items=self._batch_size,
        )
        batches = await pack_content_batches(
            coarse_batches,
            project=project,
            budget=self._budget,
            image_budget=self._image_budget,
        )
        return [
            Send(
                'image_description_worker',
                {'image_description_requests': batch},
            )
            for batch in batches
        ]

    async def worker(
        self,
        state: ImageDescriptionWorkerState,
    ) -> dict[str, list[ImageDescriptionResult]]:
        """Describe a batch with a fresh catalog and restore source positions."""
        requests = state['image_description_requests']
        builder = MultimodalContentBuilder()
        items = []
        for position, request in enumerate(requests):
            window = project_source_window(request.window, builder)
            items.append(
                ImageDescriptionInput(
                    position=position,
                    target=window.target[0].material,
                    context_before=window.context_before,
                    context_after=window.context_after,
                )
            )
        payload = (
            TypeAdapter(dict[str, object])
            .dump_json(
                {'items': items, 'content_images': []},
                serialize_as_any=True,
            )
            .decode('utf-8')
        )
        if (
            self._budget.counter.estimate(text=payload, images=builder.images)
            > self._budget.token_limit
        ):
            raise InputBudgetExceeded(
                'Source context exceeds the configured input budget'
            )
        payloads = await load_content_images(
            builder.images, budget=self._image_budget
        )
        content_images = [dspy.Image(payload) for payload in payloads]
        outputs = await self._describer.acall(
            items=items, content_images=content_images
        )
        descriptions = {
            output.position: output.description for output in outputs
        }
        return {
            'image_description_results': [
                ImageDescriptionResult(
                    flat_position=request.flat_position,
                    description=descriptions[position],
                )
                for position, request in enumerate(requests)
            ]
        }

    def collect(
        self, state: SourceProcessingState
    ) -> dict[str, list[SourcePage]]:
        """Collect descriptions while preserving source-page and block order."""
        descriptions = {
            result.flat_position: result.description
            for result in state.image_description_results
        }
        described_pages: list[SourcePage] = []
        flat_position = 0
        for page in state.image_filtered_pages:
            described_blocks: list[SourceBlock] = []
            for source_block in page.blocks:
                if (
                    source_block.block_type is BlockType.IMAGE
                    and source_block.assets
                ):
                    described_blocks.append(
                        source_block.model_copy(
                            update={
                                'content': descriptions[flat_position],
                            }
                        )
                    )
                else:
                    described_blocks.append(source_block)
                flat_position += 1
            described_pages.append(
                page.model_copy(update={'blocks': described_blocks})
            )
        return {'image_described_pages': described_pages}
