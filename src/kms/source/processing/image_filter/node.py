"""Dispatch and collect contextual source-image retention decisions."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms.config.inference import ContextWindowSettings
from kms.core.batching import pack_items
from kms.core.tokenization.budget import TokenBudget
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.inference.batching import pack_content_batches
from kms.media.budget import ImagePayloadBudget
from kms.source.context import (
    count_source_tokens,
    project_source_block,
    project_source_window,
    select_window,
)
from kms.source.models.source import BlockType, SourceBlock
from kms.source.processing.image_filter.inference import ImageFilterModule
from kms.source.processing.image_filter.models import (
    ImageFilterInput,
    ImageFilterRequest,
    ImageFilterResult,
)
from kms.source.processing.state import SourceProcessingState


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
        flat_blocks = state.image_seam_blocks
        target_positions = [
            position
            for position, block in enumerate(flat_blocks)
            if block.block_type is BlockType.IMAGE and block.image_path
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

        def project(batch: list[ImageFilterRequest]) -> dict[str, object]:
            return {'items': self._project_items(batch)}

        coarse_batches = pack_items(
            requests,
            token_counts=[1] * len(requests),
            token_budget=self._batch_size,
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
        requests: list[ImageFilterRequest],
    ) -> list[ImageFilterInput]:
        items = []
        for position, request in enumerate(requests):
            window = project_source_window(request.window)
            items.append(
                ImageFilterInput(
                    position=position,
                    target=project_source_block(request.source_block).material,
                    context_before=window.context_before,
                    context_after=window.context_after,
                )
            )
        return items

    async def worker(
        self, state: ImageFilterWorkerState
    ) -> dict[str, list[ImageFilterResult]]:
        requests = state['image_filter_requests']
        decisions = await self._module.acall(
            items=self._project_items(requests)
        )
        by_position = {
            decision.position: decision.keep for decision in decisions
        }
        return {
            'image_filter_results': [
                ImageFilterResult(
                    block_uuid=request.source_block.uuid,
                    keep=by_position[position],
                )
                for position, request in enumerate(requests)
            ]
        }

    def collect(
        self, state: SourceProcessingState
    ) -> dict[str, list[SourceBlock]]:
        """Remove rejected image targets by stable block UUID."""
        decisions = {
            result.block_uuid: result.keep
            for result in state.image_filter_results
        }
        return {
            'image_filtered_blocks': [
                block
                for block in state.image_seam_blocks
                if not (
                    block.block_type is BlockType.IMAGE
                    and block.image_path
                    and not decisions[block.uuid]
                )
            ]
        }
