"""LangGraph node for source-page text seam processing."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.core.model import BlockType, SourceBlock, SourcePage
from kms2.core.model.source_stage.text_seam import (
    TextSeamRequest,
    TextSeamResult,
)
from kms2.langgraph.source.state import SourceState
from kms2.module.source.text_seam import (
    TextSeamJudgeModule,
    TextSeamRewriterModule,
)


class TextSeamWorkerState(TypedDict):
    """State supplied to one dispatched seam worker."""

    text_seam_request: TextSeamRequest


def _skip_for_text(source_block: SourceBlock) -> bool:
    return bool(
        source_block.block_type is BlockType.BIBLIOGRAPHIC
        or source_block.block_type is BlockType.NOTE
        or source_block.block_type is BlockType.IMAGE
        or source_block.assets
    )


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
        return source_block if _text_mergeable(source_block) else None
    return None


class TextSeamNode:
    """Dispatch and collect two parity passes of page-local text seam work."""

    def __init__(
        self,
        judge: TextSeamJudgeModule,
        rewriter: TextSeamRewriterModule,
    ) -> None:
        self._judge = judge
        self._rewriter = rewriter

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
        list[Send] | Literal['text_seam_even_collect', 'text_seam_odd_collect']
    ):
        pairs = TextSeamNode._pairs(pages, parity)
        sends = [
            Send(
                worker,
                {
                    'text_seam_request': TextSeamRequest(
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
    ) -> list[Send] | Literal['text_seam_even_collect']:
        """Dispatch adjacent pairs whose upper page index is even."""
        return self._dispatch(
            state.formatted_pages,
            parity=0,
            worker='text_seam_even_worker',
            collector='text_seam_even_collect',
        )

    async def _merge_pair(self, request: TextSeamRequest) -> TextSeamResult:
        top_blocks = list(request.top_page.blocks)
        bottom_blocks = list(request.bottom_page.blocks)
        tail_index = _edge_index(top_blocks, reverse=True)
        head_index = _edge_index(bottom_blocks, reverse=False)
        if tail_index is None or head_index is None:
            return TextSeamResult(
                top_page=request.top_page.model_copy(
                    update={'blocks': top_blocks}
                ),
                bottom_page=request.bottom_page.model_copy(
                    update={'blocks': bottom_blocks}
                ),
            )

        tail = top_blocks[tail_index]
        head = bottom_blocks[head_index]
        top_context = _context_block(top_blocks, tail_index - 1, -1)
        bottom_context = _context_block(bottom_blocks, head_index + 1, 1)
        is_split = await self._judge.aforward(
            tail=tail,
            head=head,
            top_context=top_context,
            bottom_context=bottom_context,
        )
        if is_split:
            merged = await self._rewriter.aforward(
                tail=tail,
                head=head,
                top_context=top_context,
                bottom_context=bottom_context,
            )
            top_blocks[tail_index] = tail.model_copy(update={'content': merged})
            del bottom_blocks[head_index]

        return TextSeamResult(
            top_page=request.top_page.model_copy(update={'blocks': top_blocks}),
            bottom_page=request.bottom_page.model_copy(
                update={'blocks': bottom_blocks}
            ),
        )

    async def even_worker(
        self, state: TextSeamWorkerState
    ) -> dict[str, list[TextSeamResult]]:
        """Judge and optionally merge one even-pass page pair."""
        result = await self._merge_pair(state['text_seam_request'])
        return {'text_seam_even_results': [result]}

    @staticmethod
    def _collect_pages(
        pages: list[SourcePage], results: list[TextSeamResult]
    ) -> list[SourcePage]:
        replacements: dict[int, SourcePage] = {}
        for result in results:
            replacements[result.top_page.index] = result.top_page
            replacements[result.bottom_page.index] = result.bottom_page
        return [replacements.get(page.index, page) for page in pages]

    def even_collect(self, state: SourceState) -> dict[str, list[SourcePage]]:
        """Overlay even-pass pair results onto formatted pages."""
        return {
            'text_seam_even_pages': self._collect_pages(
                state.formatted_pages, state.text_seam_even_results
            )
        }

    def dispatch_odd(
        self, state: SourceState
    ) -> list[Send] | Literal['text_seam_odd_collect']:
        """Dispatch adjacent pairs whose upper page index is odd."""
        return self._dispatch(
            state.text_seam_even_pages,
            parity=1,
            worker='text_seam_odd_worker',
            collector='text_seam_odd_collect',
        )

    async def odd_worker(
        self, state: TextSeamWorkerState
    ) -> dict[str, list[TextSeamResult]]:
        """Judge and optionally merge one odd-pass page pair."""
        result = await self._merge_pair(state['text_seam_request'])
        return {'text_seam_odd_results': [result]}

    def odd_collect(self, state: SourceState) -> dict[str, list[SourcePage]]:
        """Overlay odd-pass pair results onto even-pass pages."""
        return {
            'text_seam_pages': self._collect_pages(
                state.text_seam_even_pages, state.text_seam_odd_results
            )
        }
