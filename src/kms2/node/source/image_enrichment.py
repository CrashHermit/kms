"""LangGraph node for source-image description enrichment."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.config import ContextWindowSettings
from kms2.core.context_window import select_window
from kms2.core.model import BlockType
from kms2.core.model.image_enrichment import (
    ImageEnrichmentRequest,
    ImageEnrichmentResult,
)
from kms2.core.model.source import SourceBlock, SourcePage
from kms2.langgraph.source.state import SourceState
from kms2.module.source.image_enrichment import ImageEnrichmentModule


class ImageEnrichmentWorkerState(TypedDict):
    """State supplied to one dispatched image enrichment worker."""

    image_enrichment_request: ImageEnrichmentRequest


class ImageEnrichmentNode:
    """Dispatch and collect source-image description enrichment."""

    def __init__(
        self,
        enricher: ImageEnrichmentModule,
        context_window: ContextWindowSettings,
    ) -> None:
        self._enricher = enricher
        self._context_window = context_window

    def dispatch(
        self, state: SourceState
    ) -> list[Send] | Literal['image_enrichment_collect']:
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
                    'image_enrichment_worker',
                    {
                        'image_enrichment_request': ImageEnrichmentRequest(
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
        return sends or 'image_enrichment_collect'

    async def worker(
        self,
        state: ImageEnrichmentWorkerState,
    ) -> dict[str, list[ImageEnrichmentResult]]:
        """Enrich one dispatched image block."""
        request = state['image_enrichment_request']
        description = await self._enricher.aforward(
            source_block=request.source_block,
            context_before=request.window.context_before,
            context_after=request.window.context_after,
        )
        return {
            'image_enrichment_results': [
                ImageEnrichmentResult(
                    flat_position=request.flat_position,
                    description=description,
                )
            ]
        }

    def collect(self, state: SourceState) -> dict[str, list[SourcePage]]:
        """Collect descriptions while preserving source-page and block order."""
        descriptions = {
            result.flat_position: result.description
            for result in state.image_enrichment_results
        }
        enriched_pages: list[SourcePage] = []
        flat_position = 0
        for page in state.image_seam_pages:
            enriched_blocks: list[SourceBlock] = []
            for source_block in page.blocks:
                description = descriptions.get(flat_position)
                if description is None:
                    enriched_blocks.append(source_block)
                else:
                    enriched_blocks.append(
                        source_block.model_copy(update={'content': description})
                    )
                flat_position += 1
            enriched_pages.append(
                page.model_copy(update={'blocks': enriched_blocks})
            )
        return {'image_enriched_pages': enriched_pages}
