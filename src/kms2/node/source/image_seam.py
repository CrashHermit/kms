"""LangGraph node for source-page image seam processing."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.core.model import BlockType, SourceBlock, SourcePage
from kms2.core.model.source_stage.image_seam import (
    ImageSeamRequest,
    ImageSeamResult,
)
from kms2.langgraph.source.state import SourceState
from kms2.module.source.image_seam import ImageSeamJudgeModule


class ImageSeamWorkerState(TypedDict):
    """State supplied to one dispatched image seam worker."""

    image_seam_request: ImageSeamRequest


_APPARATUS = {BlockType.BIBLIOGRAPHIC, BlockType.NOTE}


def _image_mergeable(source_block: SourceBlock) -> bool:
    """Return whether a source block is eligible for image seam judgment."""
    return bool(
        source_block.block_type is BlockType.IMAGE
        and len(source_block.assets) == 1
        and not (source_block.content or '').strip()
    )


def _edge_index(blocks: list[SourceBlock], *, reverse: bool) -> int | None:
    """Return the nearest eligible image edge, skipping apparatus blocks."""
    indices = range(len(blocks) - 1, -1, -1) if reverse else range(len(blocks))
    for index in indices:
        source_block = blocks[index]
        if source_block.block_type in _APPARATUS:
            continue
        return index if _image_mergeable(source_block) else None
    return None


class ImageSeamNode:
    """Dispatch and collect two parity passes of page-local image seam work."""

    def __init__(self, judge: ImageSeamJudgeModule) -> None:
        self._judge = judge

    @staticmethod
    def _pairs(
        pages: list[SourcePage], parity: int
    ) -> list[tuple[SourcePage, SourcePage]]:
        return [
            (pages[index], pages[index + 1])
            for index in range(len(pages) - 1)
            if pages[index].index % 2 == parity
            and _edge_index(pages[index].blocks, reverse=True) is not None
            and _edge_index(pages[index + 1].blocks, reverse=False) is not None
        ]

    @staticmethod
    def _dispatch(
        pages: list[SourcePage], parity: int, worker: str, collector: str
    ) -> (
        list[Send]
        | Literal['image_seam_even_collect', 'image_seam_odd_collect']
    ):
        pairs = ImageSeamNode._pairs(pages, parity)
        sends = [
            Send(
                worker,
                {
                    'image_seam_request': ImageSeamRequest(
                        top_page=top_page,
                        bottom_page=bottom_page,
                    )
                },
            )
            for top_page, bottom_page in pairs
        ]
        return sends or collector  # type: ignore[return-value]

    def dispatch_even(
        self, state: SourceState
    ) -> list[Send] | Literal['image_seam_even_collect']:
        """Dispatch adjacent pairs whose upper page index is even."""
        return self._dispatch(
            state.text_seam_pages,
            parity=0,
            worker='image_seam_even_worker',
            collector='image_seam_even_collect',
        )

    async def _merge_pair(self, request: ImageSeamRequest) -> ImageSeamResult:
        top_blocks = list(request.top_page.blocks)
        bottom_blocks = list(request.bottom_page.blocks)
        top_index = _edge_index(top_blocks, reverse=True)
        bottom_index = _edge_index(bottom_blocks, reverse=False)
        if top_index is None or bottom_index is None:
            return ImageSeamResult(
                top_page=request.top_page.model_copy(
                    update={'blocks': top_blocks}
                ),
                bottom_page=request.bottom_page.model_copy(
                    update={'blocks': bottom_blocks}
                ),
            )

        top_block = top_blocks[top_index]
        bottom_block = bottom_blocks[bottom_index]
        is_continuation = await self._judge.aforward(
            top_block=top_block,
            bottom_block=bottom_block,
        )
        if is_continuation:
            top_blocks[top_index] = top_block.model_copy(
                update={'assets': [*top_block.assets, *bottom_block.assets]}
            )
            del bottom_blocks[bottom_index]

        return ImageSeamResult(
            top_page=request.top_page.model_copy(update={'blocks': top_blocks}),
            bottom_page=request.bottom_page.model_copy(
                update={'blocks': bottom_blocks}
            ),
        )

    async def even_worker(
        self, state: ImageSeamWorkerState
    ) -> dict[str, list[ImageSeamResult]]:
        """Judge and optionally merge one even-pass page pair."""
        result = await self._merge_pair(state['image_seam_request'])
        return {'image_seam_even_results': [result]}

    @staticmethod
    def _collect_pages(
        pages: list[SourcePage], results: list[ImageSeamResult]
    ) -> list[SourcePage]:
        replacements: dict[int, SourcePage] = {}
        for result in results:
            replacements[result.top_page.index] = result.top_page
            replacements[result.bottom_page.index] = result.bottom_page
        return [replacements.get(page.index, page) for page in pages]

    def even_collect(self, state: SourceState) -> dict[str, list[SourcePage]]:
        """Overlay even-pass pair results onto text-seam pages."""
        return {
            'image_seam_even_pages': self._collect_pages(
                state.text_seam_pages, state.image_seam_even_results
            )
        }

    def dispatch_odd(
        self, state: SourceState
    ) -> list[Send] | Literal['image_seam_odd_collect']:
        """Dispatch adjacent pairs whose upper page index is odd."""
        return self._dispatch(
            state.image_seam_even_pages,
            parity=1,
            worker='image_seam_odd_worker',
            collector='image_seam_odd_collect',
        )

    async def odd_worker(
        self, state: ImageSeamWorkerState
    ) -> dict[str, list[ImageSeamResult]]:
        """Judge and optionally merge one odd-pass page pair."""
        result = await self._merge_pair(state['image_seam_request'])
        return {'image_seam_odd_results': [result]}

    def odd_collect(self, state: SourceState) -> dict[str, list[SourcePage]]:
        """Overlay odd-pass pair results onto even-pass pages."""
        return {
            'image_seam_pages': self._collect_pages(
                state.image_seam_even_pages, state.image_seam_odd_results
            )
        }
