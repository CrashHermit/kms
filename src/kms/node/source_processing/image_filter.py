"""Dispatch and collect contextual source-image retention decisions."""

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
from kms.core.model.source_processing.image_filter import (
    ImageFilterInput,
    ImageFilterRequest,
    ImageFilterResult,
)
from kms.core.source_context import count_source_tokens, project_source_window
from kms.core.tokenization.budget import InputBudgetExceeded, TokenBudget
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.core.windowing import (
    pack_items,
    select_window,
)
from kms.langgraph.source_processing.state import SourceProcessingState
from kms.module.source_processing.image_filter import ImageFilterModule
from kms.service.content import load_content_images, pack_content_batches


class ImageFilterWorkerState(TypedDict):
    """State supplied to one image-filter worker."""

    image_filter_requests: list[ImageFilterRequest]


class ImageFilterNode:
    """Filter clear image artifacts while preserving all source context."""

    def __init__(
        self,
        module: ImageFilterModule,
        context_window: ContextWindowSettings,
        *,
        budget: TokenBudget,
        image_budget: ImagePayloadBudget,
        batch_size: int,
        token_counters: tuple[MaterialTokenEstimator, ...],
    ) -> None:
        self._module = module
        self._context_window = context_window
        self._budget = budget
        self._image_budget = image_budget
        self._batch_size = batch_size
        self._token_counters = token_counters

    async def dispatch(
        self, state: SourceProcessingState
    ) -> list[Send] | Literal['image_filter_collect']:
        """Dispatch image targets with bounded source windows and payloads."""
        flat_blocks = [
            block for page in state.image_seam_pages for block in page.blocks
        ]
        target_positions = [
            position
            for position, block in enumerate(flat_blocks)
            if block.block_type is BlockType.IMAGE and block.assets
        ]
        if not target_positions:
            return 'image_filter_collect'

        token_counts = count_source_tokens(flat_blocks, self._token_counters)
        requests = [
            ImageFilterRequest(
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
            for position in target_positions
        ]

        def project(
            batch: list[ImageFilterRequest], builder: MultimodalContentBuilder
        ) -> dict[str, object]:
            return {
                'items': self._project_items(batch, builder),
            }

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
            Send('image_filter_worker', {'image_filter_requests': batch})
            for batch in batches
        ]

    @staticmethod
    def _project_items(
        requests: list[ImageFilterRequest], builder: MultimodalContentBuilder
    ) -> list[ImageFilterInput]:
        items = []
        for position, request in enumerate(requests):
            window = project_source_window(request.window, builder)
            items.append(
                ImageFilterInput(
                    position=position,
                    target=window.target[0].material,
                    context_before=window.context_before,
                    context_after=window.context_after,
                )
            )
        return items

    async def worker(
        self, state: ImageFilterWorkerState
    ) -> dict[str, list[ImageFilterResult]]:
        """Gate and load the exact catalog used by one indexed model call."""
        requests = state['image_filter_requests']
        builder = MultimodalContentBuilder()
        items = self._project_items(requests, builder)
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
        decisions = await self._module.acall(
            items=items, content_images=content_images
        )
        by_position = {
            decision.position: decision.keep for decision in decisions
        }
        return {
            'image_filter_results': [
                ImageFilterResult(
                    flat_position=request.flat_position,
                    keep=by_position[position],
                )
                for position, request in enumerate(requests)
            ]
        }

    def collect(
        self, state: SourceProcessingState
    ) -> dict[str, list[SourcePage]]:
        """Remove only rejected image targets and preserve page structure."""
        decisions = {
            result.flat_position: result.keep
            for result in state.image_filter_results
        }
        filtered_pages: list[SourcePage] = []
        flat_position = 0
        for page in state.image_seam_pages:
            blocks: list[SourceBlock] = []
            for block in page.blocks:
                keep = (
                    decisions[flat_position]
                    if block.block_type is BlockType.IMAGE and block.assets
                    else True
                )
                if keep:
                    blocks.append(block)
                flat_position += 1
            filtered_pages.append(page.model_copy(update={'blocks': blocks}))

        return {'image_filtered_pages': filtered_pages}
