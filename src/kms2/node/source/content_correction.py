"""LangGraph node for visual source-content correction."""

from typing import Literal, TypedDict

import dspy
from langgraph.types import Send

from kms2.core.model.content_correction import (
    ContentCorrectionRequest,
    ContentCorrectionResult,
)
from kms2.core.model.source import SourceBlock, SourcePage
from kms2.langgraph.source.state import SourceState
from kms2.module.source.content_correction import ContentCorrectorModule


class ContentCorrectionWorkerState(TypedDict):
    """State supplied to one dispatched correction worker."""

    content_correction_request: ContentCorrectionRequest


class ContentCorrectionNode:
    """Dispatch and collect visual source-content corrections."""

    def __init__(self, corrector: ContentCorrectorModule) -> None:
        self._corrector = corrector

    def dispatch(
        self, state: SourceState
    ) -> list[Send] | Literal['content_correction_collect']:
        """Dispatch one correction request per source block."""
        sends: list[Send] = []
        for page in state.ocr_pages:
            for block_position, source_block in enumerate(page.blocks):
                request = ContentCorrectionRequest(
                    page_index=page.index,
                    block_position=block_position,
                    source_block=source_block,
                )
                sends.append(
                    Send(
                        'content_correction_worker',
                        {'content_correction_request': request},
                    )
                )
        return sends or 'content_correction_collect'

    async def worker(
        self,
        state: ContentCorrectionWorkerState,
    ) -> dict[str, list[ContentCorrectionResult]]:
        """Correct one dispatched source block."""
        request = state['content_correction_request']
        source_block = request.source_block

        block_crop: dspy.Image = dspy.Image(url=source_block.crop_path)
        corrected_content = await self._corrector.acall(
            block_crop=block_crop,
            block_type=source_block.block_type,
            content=source_block.content,
        )
        result = ContentCorrectionResult(
            page_index=request.page_index,
            block_position=request.block_position,
            source_block=source_block.model_copy(
                update={'content': corrected_content}
            ),
        )
        return {'correction_results': [result]}

    def collect(
        self,
        state: SourceState,
    ) -> dict[str, list[SourcePage]]:
        """Collect corrected source blocks into ordered source pages."""
        blocks_by_page: dict[int, list[SourceBlock]] = {}
        for result in sorted(
            state.correction_results,
            key=lambda item: (item.page_index, item.block_position),
        ):
            blocks_by_page.setdefault(result.page_index, []).append(
                result.source_block
            )

        corrected_pages = [
            SourcePage(index=page_index, blocks=blocks)
            for page_index, blocks in sorted(blocks_by_page.items())
        ]
        return {'corrected_pages': corrected_pages}
