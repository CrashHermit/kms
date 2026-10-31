"""LangGraph node for visual source-content correction."""

import json
from typing import Literal, TypedDict

import dspy
from langgraph.types import Send

from kms2.core.model.block import SourceBlock
from kms2.core.model.page import SourcePage
from kms2.core.model.source_processing.content_correction import (
    ContentCorrectionInput,
    ContentCorrectionRequest,
    ContentCorrectionResult,
)
from kms2.core.windowing import TokenBudget, pack_items
from kms2.langgraph.source_processing.state import SourceProcessingState
from kms2.module.source_processing.content_correction import (
    ContentCorrectorModule,
)


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
        batch_size: int,
    ) -> None:
        self._corrector = corrector
        self._budget = budget
        self._batch_size = batch_size

    def dispatch(
        self, state: SourceProcessingState
    ) -> list[Send] | Literal['content_correction_collect']:
        """Dispatch ordered batches within text-token and item limits."""
        requests = [
            ContentCorrectionRequest(
                page_index=page.index,
                block_position=block_position,
                source_block=source_block,
            )
            for page in state.ocr_pages
            for block_position, source_block in enumerate(page.blocks)
        ]
        if not requests:
            return 'content_correction_collect'

        item_payloads = [
            json.dumps(
                {
                    'position': self._batch_size - 1,
                    'block_type': request.source_block.block_type.value,
                    'content': request.source_block.content,
                },
                ensure_ascii=False,
                separators=(',', ':'),
            )
            for request in requests
        ]
        item_token_counts = self._budget.counter.count_texts(item_payloads)
        list_overhead = self._budget.counter.count_texts(['[,]'])[0]
        token_counts = [count + list_overhead for count in item_token_counts]
        batches = pack_items(
            requests,
            token_counts=token_counts,
            token_budget=self._budget.token_limit,
            max_items=self._batch_size,
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
        block_crops = [
            dspy.Image.from_path(request.source_block.crop_path)
            for request in requests
        ]
        corrected_blocks = await self._corrector.acall(
            blocks=[
                ContentCorrectionInput(
                    position=position,
                    block_type=request.source_block.block_type,
                    content=request.source_block.content,
                )
                for position, request in enumerate(requests)
            ],
            block_crops=block_crops,
        )
        return {
            'correction_results': [
                ContentCorrectionResult(
                    page_index=requests[block.position].page_index,
                    block_position=requests[block.position].block_position,
                    source_block=requests[
                        block.position
                    ].source_block.model_copy(
                        update={'content': block.corrected_content}
                    ),
                )
                for block in corrected_blocks
            ]
        }

    def collect(
        self,
        state: SourceProcessingState,
    ) -> dict[str, list[SourcePage]]:
        """Collect corrections while preserving original pages and blocks."""
        corrected_by_location = {
            (result.page_index, result.block_position): result.source_block
            for result in state.correction_results
        }
        corrected_pages: list[SourcePage] = []
        for page in state.ocr_pages:
            corrected_blocks: list[SourceBlock] = []
            for block_position, _source_block in enumerate(page.blocks):
                corrected_blocks.append(
                    corrected_by_location[(page.index, block_position)]
                )
            corrected_pages.append(
                page.model_copy(update={'blocks': corrected_blocks})
            )
        return {'corrected_pages': corrected_pages}
