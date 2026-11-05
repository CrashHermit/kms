"""LangGraph node for source-content representation formatting."""

import json
from typing import Literal, TypedDict

from langgraph.types import Send

from kms.core.model.block import SourceBlock
from kms.core.model.block_types import BlockType
from kms.core.model.page import SourcePage
from kms.core.model.source_processing.formatting import (
    FormattingInput,
    FormattingRequest,
    FormattingResult,
)
from kms.core.tokenization.budget import TokenBudget
from kms.core.windowing import pack_items
from kms.langgraph.source_processing.state import SourceProcessingState
from kms.module.source_processing.formatting import FormatterModule


class FormattingWorkerState(TypedDict):
    """State supplied to one dispatched formatting worker."""

    formatting_requests: list[FormattingRequest]


class FormattingNode:
    """Dispatch and collect source-content representation formatting."""

    def __init__(
        self,
        formatter: FormatterModule,
        *,
        budget: TokenBudget,
        batch_size: int,
    ) -> None:
        self._formatter = formatter
        self._budget = budget
        self._batch_size = batch_size

    def dispatch(
        self, state: SourceProcessingState
    ) -> list[Send] | Literal['formatter_collect']:
        """Dispatch ordered, token-budgeted batches of non-empty text blocks."""
        directions_by_uuid = {
            result.source_block.uuid: result.cleanup_instructions
            for result in state.correction_results
        }
        requests = [
            FormattingRequest(
                page_index=page.index,
                block_position=block_position,
                content=source_block.content,
                cleanup_instructions=directions_by_uuid[source_block.uuid],
            )
            for page in state.corrected_pages
            for block_position, source_block in enumerate(page.blocks)
            if source_block.content
            and source_block.block_type is not BlockType.IMAGE
        ]
        if not requests:
            return 'formatter_collect'

        item_payloads = [
            json.dumps(
                {
                    'position': self._batch_size - 1,
                    'content': request.content,
                    'cleanup_instructions': request.cleanup_instructions,
                },
                ensure_ascii=False,
                separators=(',', ':'),
            )
            for request in requests
        ]
        item_token_counts = self._budget.counter.count_texts(item_payloads)
        list_overhead = self._budget.counter.count_texts(['[,]'])[0]
        token_counts = [
            item_count + list_overhead for item_count in item_token_counts
        ]
        batches = pack_items(
            requests,
            token_counts=token_counts,
            token_budget=self._budget.token_limit,
            max_items=self._batch_size,
        )
        return [
            Send('formatter_worker', {'formatting_requests': batch})
            for batch in batches
        ]

    async def worker(
        self,
        state: FormattingWorkerState,
    ) -> dict[str, list[FormattingResult]]:
        """Format a batch and restore each result's original source location."""
        requests = state['formatting_requests']
        formatted_blocks = await self._formatter.acall(
            blocks=[
                FormattingInput(
                    position=position,
                    content=request.content,
                    cleanup_instructions=request.cleanup_instructions,
                )
                for position, request in enumerate(requests)
            ],
        )
        formatted_by_position = {
            block.position: block.formatted_content
            for block in formatted_blocks
        }
        return {
            'formatting_results': [
                FormattingResult(
                    page_index=request.page_index,
                    block_position=request.block_position,
                    formatted_content=formatted_by_position[position],
                )
                for position, request in enumerate(requests)
            ]
        }

    def collect(
        self, state: SourceProcessingState
    ) -> dict[str, list[SourcePage]]:
        """Collect formatting results while preserving page structure."""
        formatted_by_location = {
            (result.page_index, result.block_position): result.formatted_content
            for result in state.formatting_results
        }
        formatted_pages: list[SourcePage] = []
        for page in state.corrected_pages:
            formatted_blocks: list[SourceBlock] = []
            for block_position, source_block in enumerate(page.blocks):
                if (
                    not source_block.content
                    or source_block.block_type is BlockType.IMAGE
                ):
                    formatted_blocks.append(source_block)
                    continue
                location = (page.index, block_position)
                formatted_blocks.append(
                    source_block.model_copy(
                        update={
                            'content': formatted_by_location[location],
                        }
                    )
                )
            formatted_pages.append(
                page.model_copy(update={'blocks': formatted_blocks})
            )

        return {'formatted_pages': formatted_pages}
