"""LangGraph node for source-content representation formatting."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.core.model import BlockType, SourceBlock, SourcePage
from kms2.core.model.source_stage.formatting import (
    FormattingRequest,
    FormattingResult,
)
from kms2.langgraph.source.state import SourceState
from kms2.module.source.formatting import FormatterModule


class FormattingWorkerState(TypedDict):
    """State supplied to one dispatched formatting worker."""

    formatting_request: FormattingRequest


class FormattingNode:
    """Dispatch and collect source-content representation formatting."""

    def __init__(self, formatter: FormatterModule) -> None:
        self._formatter = formatter

    def dispatch(
        self, state: SourceState
    ) -> list[Send] | Literal['formatter_collect']:
        """Dispatch one indexed formatting request per non-empty block."""
        sends: list[Send] = []
        for page in state.corrected_pages:
            for block_position, source_block in enumerate(page.blocks):
                if (
                    not source_block.content
                    or source_block.block_type is BlockType.IMAGE
                ):
                    continue
                request = FormattingRequest(
                    page_index=page.index,
                    block_position=block_position,
                    content=source_block.content,
                )
                sends.append(
                    Send(
                        'formatter_worker',
                        {'formatting_request': request},
                    )
                )

        return sends or 'formatter_collect'

    async def worker(
        self,
        state: FormattingWorkerState,
    ) -> dict[str, list[FormattingResult]]:
        """Format one dispatched source-content block."""
        request = state['formatting_request']
        formatted_content = await self._formatter.acall(
            content=request.content,
        )
        result = FormattingResult(
            page_index=request.page_index,
            block_position=request.block_position,
            formatted_content=formatted_content,
        )
        return {'formatting_results': [result]}

    def collect(self, state: SourceState) -> dict[str, list[SourcePage]]:
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
