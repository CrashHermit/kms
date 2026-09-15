"""LangGraph node for source-image descriptions."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.config import ContextWindowSettings
from kms2.core.model import BlockType, SourceBlock, SourcePage
from kms2.core.model.source_stage.image_description import (
    ImageDescriptionRequest,
    ImageDescriptionResult,
)
from kms2.core.windowing import select_window
from kms2.langgraph.source.state import SourceState
from kms2.module.source.image_description import ImageDescriptionModule


class ImageDescriptionWorkerState(TypedDict):
    """State supplied to one dispatched image description worker."""

    image_description_request: ImageDescriptionRequest


class ImageDescriptionNode:
    """Dispatch and collect source-image descriptions."""

    def __init__(
        self,
        describer: ImageDescriptionModule,
        context_window: ContextWindowSettings,
    ) -> None:
        self._describer = describer
        self._context_window = context_window

    def dispatch(
        self, state: SourceState
    ) -> list[Send] | Literal['image_description_collect']:
        """Dispatch one context-window request per image block."""
        flat_blocks = [
            source_block
            for page in state.image_seam_pages
            for source_block in page.blocks
        ]
        sends: list[Send] = []
        for flat_position, source_block in enumerate(flat_blocks):
            if source_block.block_type is not BlockType.IMAGE:
                continue
            sends.append(
                Send(
                    'image_description_worker',
                    {
                        'image_description_request': ImageDescriptionRequest(
                            flat_position=flat_position,
                            source_block=source_block,
                            window=select_window(
                                flat_blocks,
                                [flat_position],
                                backward_budget=self._context_window.backward_budget,
                                forward_budget=self._context_window.forward_budget,
                            ),
                        )
                    },
                )
            )
        return sends or 'image_description_collect'

    async def worker(
        self,
        state: ImageDescriptionWorkerState,
    ) -> dict[str, list[ImageDescriptionResult]]:
        """Describe one dispatched image block."""
        request = state['image_description_request']
        description = await self._describer.aforward(
            source_block=request.source_block,
            context_before=request.window.context_before,
            context_after=request.window.context_after,
        )
        return {
            'image_description_results': [
                ImageDescriptionResult(
                    flat_position=request.flat_position,
                    description=description,
                )
            ]
        }

    def collect(self, state: SourceState) -> dict[str, list[SourcePage]]:
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
