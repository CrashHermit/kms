"""LangGraph node for source-page image seam processing."""

import json
from typing import Literal, TypedDict, cast

import dspy
from langgraph.types import Send

from kms2.core.model.block import SourceBlock
from kms2.core.model.block_types import BlockType
from kms2.core.model.page import SourcePage
from kms2.core.model.source_processing.image_seam import (
    ImageSeamInput,
    ImageSeamRequest,
    ImageSeamResult,
)
from kms2.core.windowing import TokenBudget, pack_items
from kms2.langgraph.source_processing.state import SourceProcessingState
from kms2.module.source_processing.image_seam import ImageSeamJudgeModule


class ImageSeamWorkerState(TypedDict):
    """State supplied to one dispatched image seam worker."""

    image_seam_requests: list[ImageSeamRequest]


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

    def __init__(
        self,
        judge: ImageSeamJudgeModule,
        *,
        budget: TokenBudget,
        batch_size: int,
    ) -> None:
        self._judge = judge
        self._budget = budget
        self._batch_size = batch_size

    @staticmethod
    def _pairs(
        pages: list[SourcePage], parity: int
    ) -> list[tuple[SourcePage, SourcePage]]:
        return [
            (pages[index], pages[index + 1])
            for index in range(len(pages) - 1)
            if pages[index + 1].index == pages[index].index + 1
            and pages[index].index % 2 == parity
            and _edge_index(pages[index].blocks, reverse=True) is not None
            and _edge_index(pages[index + 1].blocks, reverse=False) is not None
        ]

    @staticmethod
    def _edge_positions(request: ImageSeamRequest) -> tuple[int, int]:
        top_position = cast(
            int, _edge_index(request.top_page.blocks, reverse=True)
        )
        bottom_position = cast(
            int, _edge_index(request.bottom_page.blocks, reverse=False)
        )
        return top_position, bottom_position

    def _payload_costs(self, requests: list[ImageSeamRequest]) -> list[int]:
        items = [
            ImageSeamInput(
                position=self._batch_size - 1,
                top_image_position=2 * (self._batch_size - 1),
                bottom_image_position=2 * (self._batch_size - 1) + 1,
            )
            for _request in requests
        ]
        payloads = [
            json.dumps(
                item.model_dump(mode='json'),
                ensure_ascii=False,
                separators=(',', ':'),
            )
            for item in items
        ]
        costs = self._budget.counter.count_texts(payloads)
        list_overhead = self._budget.counter.count_texts(['[,]'])[0]
        return [cost + list_overhead for cost in costs]

    def _dispatch(
        self, pages: list[SourcePage], parity: int, worker: str, collector: str
    ) -> (
        list[Send]
        | Literal['image_seam_even_collect', 'image_seam_odd_collect']
    ):
        requests = [
            ImageSeamRequest(top_page=top, bottom_page=bottom)
            for top, bottom in self._pairs(pages, parity)
        ]
        if not requests:
            return collector  # type: ignore[return-value]
        envelope_cost = self._budget.counter.count_texts(
            ['{"items":[],"seam_images":[]}']
        )[0]
        batches = pack_items(
            requests,
            token_counts=self._payload_costs(requests),
            token_budget=self._budget.token_limit - envelope_cost,
            max_items=self._batch_size,
        )
        return [
            Send(worker, {'image_seam_requests': batch}) for batch in batches
        ]

    def dispatch_even(
        self, state: SourceProcessingState
    ) -> list[Send] | Literal['image_seam_even_collect']:
        """Dispatch adjacent pairs whose upper page index is even."""
        return self._dispatch(
            state.text_seam_pages,
            0,
            'image_seam_even_worker',
            'image_seam_even_collect',
        )

    async def _merge_requests(
        self, requests: list[ImageSeamRequest]
    ) -> list[ImageSeamResult]:
        edge_positions = [self._edge_positions(request) for request in requests]
        seam_images = [
            dspy.Image.from_path(asset.path)
            for request, (top_position, bottom_position) in zip(
                requests, edge_positions, strict=True
            )
            for page, block_position in (
                (request.top_page, top_position),
                (request.bottom_page, bottom_position),
            )
            for asset in page.blocks[block_position].assets
        ]
        items = [
            ImageSeamInput(
                position=position,
                top_image_position=2 * position,
                bottom_image_position=2 * position + 1,
            )
            for position, _request in enumerate(requests)
        ]
        decisions = await self._judge.acall(
            items=items,
            seam_images=seam_images,
        )
        continuation_by_position = {
            decision.position: decision.is_continuation
            for decision in decisions
        }
        results: list[ImageSeamResult] = []
        for position, request in enumerate(requests):
            top_blocks = list(request.top_page.blocks)
            bottom_blocks = list(request.bottom_page.blocks)
            if continuation_by_position[position]:
                top_index, bottom_index = self._edge_positions(request)
                top_block = top_blocks[top_index]
                bottom_block = bottom_blocks[bottom_index]
                top_blocks[top_index] = top_block.model_copy(
                    update={'assets': [*top_block.assets, *bottom_block.assets]}
                )
                del bottom_blocks[bottom_index]
            results.append(
                ImageSeamResult(
                    top_page=request.top_page.model_copy(
                        update={'blocks': top_blocks}
                    ),
                    bottom_page=request.bottom_page.model_copy(
                        update={'blocks': bottom_blocks}
                    ),
                )
            )
        return results

    async def even_worker(
        self, state: ImageSeamWorkerState
    ) -> dict[str, list[ImageSeamResult]]:
        """Judge and merge one even-pass packet of page pairs."""
        return {
            'image_seam_even_results': await self._merge_requests(
                state['image_seam_requests']
            )
        }

    @staticmethod
    def _collect_pages(
        pages: list[SourcePage], results: list[ImageSeamResult]
    ) -> list[SourcePage]:
        replacements: dict[int, SourcePage] = {}
        for result in results:
            replacements[result.top_page.index] = result.top_page
            replacements[result.bottom_page.index] = result.bottom_page
        return [replacements.get(page.index, page) for page in pages]

    def even_collect(
        self, state: SourceProcessingState
    ) -> dict[str, list[SourcePage]]:
        """Overlay even-pass pair results onto text-seam pages."""
        return {
            'image_seam_even_pages': self._collect_pages(
                state.text_seam_pages, state.image_seam_even_results
            )
        }

    def dispatch_odd(
        self, state: SourceProcessingState
    ) -> list[Send] | Literal['image_seam_odd_collect']:
        """Dispatch adjacent pairs whose upper page index is odd."""
        return self._dispatch(
            state.image_seam_even_pages,
            1,
            'image_seam_odd_worker',
            'image_seam_odd_collect',
        )

    async def odd_worker(
        self, state: ImageSeamWorkerState
    ) -> dict[str, list[ImageSeamResult]]:
        """Judge and merge one odd-pass packet of page pairs."""
        return {
            'image_seam_odd_results': await self._merge_requests(
                state['image_seam_requests']
            )
        }

    def odd_collect(
        self, state: SourceProcessingState
    ) -> dict[str, list[SourcePage]]:
        """Overlay odd-pass pair results onto even-pass pages."""
        return {
            'image_seam_pages': self._collect_pages(
                state.image_seam_even_pages, state.image_seam_odd_results
            )
        }
