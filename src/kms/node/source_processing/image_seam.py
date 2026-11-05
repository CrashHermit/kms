"""LangGraph node for source-page image seam processing."""

from typing import Literal, TypedDict, cast

import dspy
from langgraph.types import Send
from pydantic import TypeAdapter

from kms.core.content import MultimodalContentBuilder
from kms.core.media_budget import ImagePayloadBudget
from kms.core.model.block import SourceBlock
from kms.core.model.block_types import BlockType
from kms.core.model.page import SourcePage
from kms.core.model.source_processing.image_seam import (
    ImageSeamInput,
    ImageSeamRequest,
    ImageSeamResult,
)
from kms.core.source_context import project_source_block
from kms.core.tokenization.budget import InputBudgetExceeded, TokenBudget
from kms.core.windowing import pack_items
from kms.langgraph.source_processing.state import SourceProcessingState
from kms.module.source_processing.image_seam import ImageSeamJudgeModule
from kms.service.content import load_content_images, pack_content_batches


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
        image_budget: ImagePayloadBudget,
        batch_size: int,
    ) -> None:
        self._judge = judge
        self._budget = budget
        self._image_budget = image_budget
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

    def _project(
        self,
        requests: list[ImageSeamRequest],
        builder: MultimodalContentBuilder,
    ) -> dict[str, object]:
        items = []
        for position, request in enumerate(requests):
            top_position, bottom_position = self._edge_positions(request)
            top_block = project_source_block(
                request.top_page.blocks[top_position], builder
            )
            bottom_block = project_source_block(
                request.bottom_page.blocks[bottom_position], builder
            )
            items.append(
                ImageSeamInput(
                    position=position,
                    top=builder.build(
                        text=None,
                        images=[
                            builder.images[index]
                            for index in top_block.material.image_positions
                        ],
                    ),
                    bottom=builder.build(
                        text=None,
                        images=[
                            builder.images[index]
                            for index in bottom_block.material.image_positions
                        ],
                    ),
                )
            )
        return {'items': items}

    async def _dispatch(
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
            ['{"items":[],"content_images":[]}']
        )[0]
        coarse_batches = pack_items(
            requests,
            token_counts=[1] * len(requests),
            token_budget=max(self._budget.token_limit - envelope_cost, 1),
            max_items=self._batch_size,
        )
        batches = await pack_content_batches(
            coarse_batches,
            project=self._project,
            budget=self._budget,
            image_budget=self._image_budget,
        )
        return [
            Send(worker, {'image_seam_requests': batch}) for batch in batches
        ]

    async def dispatch_even(
        self, state: SourceProcessingState
    ) -> list[Send] | Literal['image_seam_even_collect']:
        """Dispatch adjacent pairs whose upper page index is even."""
        return await self._dispatch(
            state.text_seam_pages,
            0,
            'image_seam_even_worker',
            'image_seam_even_collect',
        )

    async def _merge_requests(
        self, requests: list[ImageSeamRequest]
    ) -> list[ImageSeamResult]:
        builder = MultimodalContentBuilder()
        items = cast(
            list[ImageSeamInput], self._project(requests, builder)['items']
        )
        payload = (
            TypeAdapter(dict[str, object])
            .dump_json(
                {'items': items, 'content_images': []},
                serialize_as_any=True,
            )
            .decode('utf-8')
        )
        if (
            self._budget.counter.estimate(text=payload, images=builder.images)
            > self._budget.token_limit
        ):
            raise InputBudgetExceeded(
                'Source context exceeds the configured input budget'
            )
        payloads = await load_content_images(
            builder.images, budget=self._image_budget
        )
        content_images = [dspy.Image(payload) for payload in payloads]
        decisions = await self._judge.acall(
            items=items, content_images=content_images
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

    async def dispatch_odd(
        self, state: SourceProcessingState
    ) -> list[Send] | Literal['image_seam_odd_collect']:
        """Dispatch adjacent pairs whose upper page index is odd."""
        return await self._dispatch(
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
