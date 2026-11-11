"""LangGraph node for visual source-content correction."""

import asyncio
import hashlib
from pathlib import Path
from typing import Literal, TypedDict, cast

from langgraph.types import Send
from PIL import Image

from kms.core.batching import pack_items
from kms.core.tokenization.budget import TokenBudget
from kms.inference.batching import pack_content_batches
from kms.media.budget import ImagePayloadBudget
from kms.media.models import ImageAsset, ImagePart, Material, TextPart
from kms.source.images import stitch_positioned_crops
from kms.source.models.source import BlockType, SourceBlock, SourcePage
from kms.source.processing.correction.inference import (
    ContentCorrectorModule,
)
from kms.source.processing.correction.models import (
    ContentCorrectionBlockContext,
    ContentCorrectionInput,
    ContentCorrectionRequest,
    ContentCorrectionResult,
)
from kms.source.processing.state import SourceProcessingState


class ContentCorrectionWorkerState(TypedDict):
    """State supplied to one dispatched correction worker."""

    content_correction_requests: list[ContentCorrectionRequest]


class ContentCorrectionNode:
    """Dispatch and collect visual source-content corrections."""

    def __init__(
        self,
        corrector: ContentCorrectorModule,
        *,
        budget: TokenBudget,
        image_budget: ImagePayloadBudget,
        batch_size: int,
    ) -> None:
        self._corrector = corrector
        self._budget = budget
        self._image_budget = image_budget
        self._batch_size = batch_size

    def _requests(
        self, state: SourceProcessingState
    ) -> list[ContentCorrectionRequest]:
        """Build one positioned neighboring-block image per text target."""
        page_offsets: dict[int, int] = {}
        page_groups: dict[int, int] = {}
        offset = 0
        group = 0
        previous_index = None
        for page in state.ocr_pages:
            if previous_index is not None and page.index != previous_index + 1:
                offset = 0
                group += 1
            page_offsets[page.index] = offset
            page_groups[page.index] = group
            with Image.open(state.ocr_page_images[page.index]) as image:
                offset += image.height
            previous_index = page.index

        blocks = [
            (page.index, position, block)
            for page in state.ocr_pages
            for position, block in enumerate(page.blocks)
        ]
        requests = []
        for index, (page_index, position, block) in enumerate(blocks):
            if block.block_type is BlockType.IMAGE:
                continue
            neighbors = [
                item
                for item in blocks[max(0, index - 1) : index + 2]
                if page_groups[item[0]] == page_groups[page_index]
            ]
            target_index = next(
                i
                for i, item in enumerate(neighbors)
                if item[0] == page_index and item[1] == position
            )
            payload, bounds = stitch_positioned_crops(
                [
                    (
                        Path(state.ocr_page_images[neighbor_page]),
                        cast(tuple[int, int, int, int], neighbor.crop_bbox),
                        page_offsets[neighbor_page],
                    )
                    for neighbor_page, _, neighbor in neighbors
                ],
                budget=self._image_budget,
            )
            requests.append(
                ContentCorrectionRequest(
                    page_index=page_index,
                    block_position=position,
                    source_block=block,
                    context_image=payload,
                    context_digest=hashlib.sha256(payload).hexdigest(),
                    target_bounds=bounds[target_index],
                )
            )
        return requests

    @staticmethod
    def _build_inputs(
        requests: list[ContentCorrectionRequest],
    ) -> list[ContentCorrectionInput]:
        inputs = []
        for position, request in enumerate(requests):
            asset = ImageAsset(
                key=('path', request.source_block.crop_path),
                payload=Path(request.source_block.crop_path),
            )
            parts = (
                (TextPart(text=request.source_block.content),)
                if request.source_block.content is not None
                else ()
            ) + (ImagePart(asset=asset),)
            inputs.append(
                ContentCorrectionInput(
                    position=position,
                    block_type=request.source_block.block_type,
                    material=Material(parts=parts),
                    context=ContentCorrectionBlockContext(
                        image=ImageAsset(
                            key=(
                                'correction-block-context',
                                request.context_digest,
                            ),
                            payload=request.context_image,
                        ),
                        target_bounds=request.target_bounds,
                    ),
                )
            )
        return inputs

    @staticmethod
    def _project(requests: list[ContentCorrectionRequest]) -> dict[str, object]:
        return {'blocks': ContentCorrectionNode._build_inputs(requests)}

    async def dispatch(
        self, state: SourceProcessingState
    ) -> list[Send] | Literal['content_correction_collect']:
        """Dispatch non-image crops in whole-item batches."""
        if not any(
            block.block_type is not BlockType.IMAGE
            for page in state.ocr_pages
            for block in page.blocks
        ):
            return 'content_correction_collect'
        requests = await asyncio.to_thread(self._requests, state)
        if not requests:
            return 'content_correction_collect'

        coarse_batches = pack_items(
            requests,
            token_counts=[1] * len(requests),
            token_budget=self._batch_size,
            max_items=self._batch_size,
        )
        batches = await pack_content_batches(
            coarse_batches,
            project=self._project,
            budget=self._budget,
            image_budget=self._image_budget,
        )
        return [
            Send(
                'content_correction_worker',
                {'content_correction_requests': batch},
            )
            for batch in batches
        ]

    async def worker(
        self,
        state: ContentCorrectionWorkerState,
    ) -> dict[str, list[ContentCorrectionResult]]:
        """Correct a batch and restore each result's original source location."""
        requests = state['content_correction_requests']
        blocks = self._build_inputs(requests)
        corrected_blocks = await self._corrector.acall(blocks=blocks)
        corrected_by_position = {
            block.position: block for block in corrected_blocks
        }
        return {
            'correction_results': [
                ContentCorrectionResult(
                    page_index=request.page_index,
                    block_position=request.block_position,
                    source_block=request.source_block.model_copy(
                        update={
                            'content': corrected_by_position[
                                position
                            ].corrected_content
                        }
                    ),
                    keep=corrected_by_position[position].keep,
                    cleanup_instructions=corrected_by_position[
                        position
                    ].cleanup_instructions,
                )
                for position, request in enumerate(requests)
            ]
        }

    def collect(
        self,
        state: SourceProcessingState,
    ) -> dict[str, list[SourcePage]]:
        """Apply explicit rejection while preserving original page structure."""
        corrected_by_location = {
            (result.page_index, result.block_position): result
            for result in state.correction_results
        }
        corrected_pages: list[SourcePage] = []
        for page in state.ocr_pages:
            corrected_blocks: list[SourceBlock] = []
            for block_position, source_block in enumerate(page.blocks):
                if source_block.block_type is BlockType.IMAGE:
                    corrected_blocks.append(source_block)
                    continue
                result = corrected_by_location[(page.index, block_position)]
                if result.keep:
                    corrected_blocks.append(result.source_block)
            corrected_pages.append(
                page.model_copy(update={'blocks': corrected_blocks})
            )
        return {'corrected_pages': corrected_pages}
