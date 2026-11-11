"""LangGraph node for source-content representation formatting."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms.core.tokenization.budget import TokenBudget
from kms.inference.batching import pack_content_batches
from kms.source.models.source import BlockType, SourceBlock, SourcePage
from kms.source.processing.formatting.inference import FormatterModule
from kms.source.processing.formatting.models import (
    FormattingInput,
    FormattingRequest,
    FormattingResult,
)
from kms.source.processing.state import SourceProcessingState


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

    async def dispatch(
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

        batches = [
            requests[start : start + self._batch_size]
            for start in range(0, len(requests), self._batch_size)
        ]
        batches = await pack_content_batches(
            batches,
            project=self._project,
            budget=self._budget,
            image_budget=None,
        )
        return [
            Send('formatter_worker', {'formatting_requests': batch})
            for batch in batches
        ]

    @staticmethod
    def _project(requests: list[FormattingRequest]) -> dict[str, object]:
        return {
            'blocks': [
                FormattingInput(
                    position=position,
                    content=request.content,
                    cleanup_instructions=request.cleanup_instructions,
                )
                for position, request in enumerate(requests)
            ]
        }

    async def worker(
        self,
        state: FormattingWorkerState,
    ) -> dict[str, list[FormattingResult]]:
        """Format a batch and restore each result's original source location."""
        requests = state['formatting_requests']
        formatted_blocks = await self._formatter.acall(
            **self._project(requests)
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
