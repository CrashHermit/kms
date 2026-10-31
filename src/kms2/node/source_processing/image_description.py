"""LangGraph node for source-image descriptions."""

import json
from typing import Literal, TypedDict

import dspy
from langgraph.types import Send

from kms2.config.inference import ContextWindowSettings
from kms2.core.model.block import SourceBlock
from kms2.core.model.block_types import BlockType
from kms2.core.model.page import SourcePage
from kms2.core.model.source_processing.image_description import (
    ImageDescriptionInput,
    ImageDescriptionRequest,
    ImageDescriptionResult,
)
from kms2.core.windowing import (
    TextTokenCounter,
    TokenBudget,
    count_text_tokens,
    pack_items,
    select_window,
)
from kms2.langgraph.source_processing.state import SourceProcessingState
from kms2.module.source_processing.image_description import (
    ImageDescriptionModule,
)


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
        batch_size: int,
        token_counters: tuple[TextTokenCounter, ...],
    ) -> None:
        self._describer = describer
        self._context_window = context_window
        self._budget = budget
        self._batch_size = batch_size
        self._token_counters = token_counters

    async def dispatch(
        self, state: SourceProcessingState
    ) -> list[Send] | Literal['image_description_collect']:
        """Dispatch token-bounded batches of logical image requests."""
        flat_blocks = [
            source_block
            for page in state.image_seam_pages
            for source_block in page.blocks
        ]
        image_positions = [
            position
            for position, block in enumerate(flat_blocks)
            if block.block_type is BlockType.IMAGE
        ]
        if not image_positions:
            return 'image_description_collect'
        token_counts = count_text_tokens(
            [block.content for block in flat_blocks], self._token_counters
        )
        requests = [
            ImageDescriptionRequest(
                flat_position=flat_position,
                source_block=flat_blocks[flat_position],
                window=select_window(
                    flat_blocks,
                    [flat_position],
                    token_counts=token_counts,
                    backward_budget=self._context_window.backward_budget,
                    forward_budget=self._context_window.forward_budget,
                ),
            )
            for flat_position in image_positions
        ]
        total_assets = sum(
            len(request.source_block.assets) for request in requests
        )
        max_image_index = max(total_assets - 1, 0)
        item_payloads = []
        for request in requests:
            item = ImageDescriptionInput(
                position=self._batch_size - 1,
                image_positions=(
                    [max_image_index] * len(request.source_block.assets)
                ),
                context_before=request.window.context_before,
                context_after=request.window.context_after,
            )
            item_payloads.append(
                json.dumps(
                    item.model_dump(mode='json'),
                    ensure_ascii=False,
                    separators=(',', ':'),
                )
            )
        item_costs = self._budget.counter.count_texts(item_payloads)
        list_overhead = self._budget.counter.count_texts(['[,]'])[0]
        envelope_cost = self._budget.counter.count_texts(
            ['{"items":[],"target_images":[]}']
        )[0]
        batches = pack_items(
            requests,
            token_counts=[cost + list_overhead for cost in item_costs],
            token_budget=self._budget.token_limit - envelope_cost,
            max_items=self._batch_size,
        )
        sends: list[Send] = []
        for batch in batches:
            sends.append(
                Send(
                    'image_description_worker',
                    {'image_description_requests': batch},
                )
            )
        return sends

    async def worker(
        self,
        state: ImageDescriptionWorkerState,
    ) -> dict[str, list[ImageDescriptionResult]]:
        """Describe a batch of logical images and restore source positions."""
        requests = state['image_description_requests']
        target_images = [
            dspy.Image.from_path(asset.path)
            for request in requests
            for asset in request.source_block.assets
        ]
        items = []
        image_position = 0
        for position, request in enumerate(requests):
            asset_count = len(request.source_block.assets)
            items.append(
                ImageDescriptionInput(
                    position=position,
                    image_positions=list(
                        range(image_position, image_position + asset_count)
                    ),
                    context_before=request.window.context_before,
                    context_after=request.window.context_after,
                )
            )
            image_position += asset_count
        outputs = await self._describer.acall(
            items=items,
            target_images=target_images,
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
        for page in state.image_seam_pages:
            described_blocks: list[SourceBlock] = []
            for source_block in page.blocks:
                description = descriptions.get(flat_position)
                if description is None:
                    described_blocks.append(source_block)
                else:
                    described_blocks.append(
                        source_block.model_copy(update={'content': description})
                    )
                flat_position += 1
            described_pages.append(
                page.model_copy(update={'blocks': described_blocks})
            )
        return {'image_described_pages': described_pages}
