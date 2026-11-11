"""LangGraph node for source-page text seam processing."""

from typing import Literal, TypedDict, cast

from langgraph.types import Send

from kms.core.tokenization.budget import TokenBudget
from kms.inference.batching import pack_content_batches
from kms.source.context import project_source_block
from kms.source.models.source import BlockType, SourceBlock, SourcePage
from kms.source.processing.state import SourceProcessingState
from kms.source.processing.text_seam.inference import (
    TextSeamJudgeModule,
    TextSeamRewriterModule,
)
from kms.source.processing.text_seam.models import (
    TextSeamInput,
    TextSeamRequest,
    TextSeamResult,
)


class TextSeamWorkerState(TypedDict):
    """State supplied to one dispatched seam worker."""

    text_seam_requests: list[TextSeamRequest]


def _skip_for_text(source_block: SourceBlock) -> bool:
    return source_block.block_type in {
        BlockType.BIBLIOGRAPHIC,
        BlockType.NOTE,
        BlockType.IMAGE,
    }


def _text_mergeable(source_block: SourceBlock) -> bool:
    return bool(source_block.content and not _skip_for_text(source_block))


def _edge_index(blocks: list[SourceBlock], *, reverse: bool) -> int | None:
    indices = range(len(blocks) - 1, -1, -1) if reverse else range(len(blocks))
    for index in indices:
        source_block = blocks[index]
        if _skip_for_text(source_block):
            continue
        return index if _text_mergeable(source_block) else None
    return None


def _context_block(
    blocks: list[SourceBlock], start: int, step: int
) -> SourceBlock | None:
    for index in range(start, len(blocks) if step > 0 else -1, step):
        source_block = blocks[index]
        if _skip_for_text(source_block):
            continue
        if source_block.content:
            return source_block
    return None


def _request_input(
    request: TextSeamRequest, position: int
) -> tuple[TextSeamInput, int, int]:
    top_blocks = request.top_page.blocks
    bottom_blocks = request.bottom_page.blocks
    tail_index = cast(int, _edge_index(top_blocks, reverse=True))
    head_index = cast(int, _edge_index(bottom_blocks, reverse=False))
    top_context = _context_block(top_blocks, tail_index - 1, -1)
    bottom_context = _context_block(bottom_blocks, head_index + 1, 1)
    tail = project_source_block(top_blocks[tail_index])
    head = project_source_block(bottom_blocks[head_index])
    projected_top_context = (
        project_source_block(top_context) if top_context else None
    )
    projected_bottom_context = (
        project_source_block(bottom_context) if bottom_context else None
    )
    return (
        TextSeamInput(
            position=position,
            tail=tail,
            head=head,
            top_context=projected_top_context,
            bottom_context=projected_bottom_context,
        ),
        tail_index,
        head_index,
    )


class TextSeamNode:
    """Dispatch and collect two parity passes of page-local text seam work."""

    def __init__(
        self,
        judge: TextSeamJudgeModule,
        rewriter: TextSeamRewriterModule,
        *,
        judge_budget: TokenBudget,
        rewrite_budget: TokenBudget,
        judge_batch_size: int,
        rewrite_batch_size: int,
    ) -> None:
        self._judge = judge
        self._rewriter = rewriter
        self._judge_budget = judge_budget
        self._rewrite_budget = rewrite_budget
        self._judge_batch_size = judge_batch_size
        self._rewrite_batch_size = rewrite_batch_size

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

    async def _dispatch(
        self, pages: list[SourcePage], parity: int, worker: str, collector: str
    ) -> (
        list[Send] | Literal['text_seam_even_collect', 'text_seam_odd_collect']
    ):
        requests = [
            TextSeamRequest(top_page=top, bottom_page=bottom)
            for top, bottom in self._pairs(pages, parity)
        ]
        if not requests:
            return collector  # type: ignore[return-value]

        def project(batch: list[TextSeamRequest]) -> dict[str, object]:
            return {
                'items': [
                    _request_input(request, position)[0]
                    for position, request in enumerate(batch)
                ]
            }

        incoming = [
            requests[position : position + self._judge_batch_size]
            for position in range(0, len(requests), self._judge_batch_size)
        ]
        batches = await pack_content_batches(
            incoming,
            project=project,
            budget=self._judge_budget,
            image_budget=None,
        )
        return [
            Send(worker, {'text_seam_requests': batch}) for batch in batches
        ]

    async def dispatch_even(
        self, state: SourceProcessingState
    ) -> list[Send] | Literal['text_seam_even_collect']:
        """Dispatch adjacent pairs whose upper page index is even."""
        return await self._dispatch(
            state.formatted_pages,
            0,
            'text_seam_even_worker',
            'text_seam_even_collect',
        )

    async def _merge_requests(
        self, requests: list[TextSeamRequest]
    ) -> list[TextSeamResult]:
        items = [
            _request_input(request, position)[0]
            for position, request in enumerate(requests)
        ]
        decisions = await self._judge.acall(items=items)
        split_positions = {
            decision.position for decision in decisions if decision.is_split
        }
        rewrites_by_position: dict[int, str] = {}
        if split_positions:
            positive_items = [
                item for item in items if item.position in split_positions
            ]
            incoming = [
                positive_items[position : position + self._rewrite_batch_size]
                for position in range(
                    0, len(positive_items), self._rewrite_batch_size
                )
            ]
            batches = await pack_content_batches(
                incoming,
                project=lambda batch: {'items': batch},
                budget=self._rewrite_budget,
                image_budget=None,
            )
            for batch in batches:
                rewrites = await self._rewriter.acall(items=batch)
                rewrites_by_position.update(
                    (rewrite.position, rewrite.merged) for rewrite in rewrites
                )
        results = []
        for position, request in enumerate(requests):
            top_blocks = list(request.top_page.blocks)
            bottom_blocks = list(request.bottom_page.blocks)
            if position in split_positions:
                _item, tail_index, head_index = _request_input(
                    request, position
                )
                tail = top_blocks[tail_index]
                top_blocks[tail_index] = tail.model_copy(
                    update={'content': rewrites_by_position[position]}
                )
                del bottom_blocks[head_index]
            results.append(
                TextSeamResult(
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
        self, state: TextSeamWorkerState
    ) -> dict[str, list[TextSeamResult]]:
        """Judge and rewrite one even-pass packet of page pairs."""
        return {
            'text_seam_even_results': await self._merge_requests(
                state['text_seam_requests']
            )
        }

    @staticmethod
    def _collect_pages(
        pages: list[SourcePage], results: list[TextSeamResult]
    ) -> list[SourcePage]:
        replacements: dict[int, SourcePage] = {}
        for result in results:
            replacements[result.top_page.index] = result.top_page
            replacements[result.bottom_page.index] = result.bottom_page
        return [replacements.get(page.index, page) for page in pages]

    def even_collect(
        self, state: SourceProcessingState
    ) -> dict[str, list[SourcePage]]:
        """Overlay even-pass pair results onto formatted pages."""
        return {
            'text_seam_even_pages': self._collect_pages(
                state.formatted_pages, state.text_seam_even_results
            )
        }

    async def dispatch_odd(
        self, state: SourceProcessingState
    ) -> list[Send] | Literal['text_seam_odd_collect']:
        """Dispatch adjacent pairs whose upper page index is odd."""
        return await self._dispatch(
            state.text_seam_even_pages,
            1,
            'text_seam_odd_worker',
            'text_seam_odd_collect',
        )

    async def odd_worker(
        self, state: TextSeamWorkerState
    ) -> dict[str, list[TextSeamResult]]:
        """Judge and rewrite one odd-pass packet of page pairs."""
        return {
            'text_seam_odd_results': await self._merge_requests(
                state['text_seam_requests']
            )
        }

    def odd_collect(
        self, state: SourceProcessingState
    ) -> dict[str, list[SourcePage]]:
        """Overlay odd-pass pair results onto even-pass pages."""
        return {
            'text_seam_pages': self._collect_pages(
                state.text_seam_even_pages, state.text_seam_odd_results
            )
        }
