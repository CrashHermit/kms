"""LangGraph node for routed source-block splitting."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.config import ContextWindowSettings
from kms2.core.model import (
    SourceBlock,
    SourcePage,
    SplitCandidate,
    SplitRequest,
    SplitResult,
)
from kms2.core.windowing import select_window
from kms2.langgraph.source.state import SourceState
from kms2.module.source.splitter import (
    ExerciseSplitterModule,
    ExerciseStripRouterModule,
)


class SplitterWorkerState(TypedDict):
    """State supplied to one dispatched splitter worker."""

    split_request: SplitRequest


class SplitterNode:
    """Dispatch, route, split, and collect source blocks."""

    def __init__(
        self,
        router: ExerciseStripRouterModule,
        splitter: ExerciseSplitterModule,
        context_window: ContextWindowSettings,
    ) -> None:
        self._router = router
        self._splitter = splitter
        self._context_window = context_window

    def dispatch(
        self, state: SourceState
    ) -> list[Send] | Literal['splitter_collect']:
        """Dispatch one context-window request per source block."""
        flat_blocks = [
            source_block
            for page in state.image_described_pages
            for source_block in page.blocks
        ]
        sends: list[Send] = []
        flat_position = 0
        while flat_position < len(flat_blocks):
            window = select_window(
                flat_blocks,
                [flat_position],
                backward_budget=self._context_window.backward_budget,
                forward_budget=self._context_window.forward_budget,
                target_budget=self._context_window.target_budget,
            )
            if not window.target:
                raise RuntimeError('splitter selected an empty target window')
            sends.append(
                Send(
                    'splitter_worker',
                    {
                        'split_request': SplitRequest(
                            flat_position=flat_position,
                            window=window,
                        )
                    },
                )
            )
            flat_position += len(window.target)
        return sends or 'splitter_collect'

    async def worker(
        self,
        state: SplitterWorkerState,
    ) -> dict[str, list[SplitResult]]:
        """Route and split one dispatched source-block window."""
        request = state['split_request']
        window = request.window
        candidates: list[SplitCandidate] = []
        for target_position, target_block in enumerate(window.target):
            contains_multiple = await self._router.aforward(
                context_before=(
                    window.context_before + window.target[:target_position]
                ),
                target_block=target_block,
                context_after=(
                    window.target[target_position + 1 :] + window.context_after
                ),
            )
            if contains_multiple:
                candidates.append(
                    SplitCandidate(
                        position=target_position,
                        source_block=target_block,
                    )
                )
        if not candidates:
            return {
                'split_results': [
                    SplitResult(flat_position=request.flat_position)
                ]
            }

        decisions = await self._splitter.aforward(
            context_before=window.context_before,
            candidates=candidates,
            context_after=window.context_after,
        )
        return {
            'split_results': [
                SplitResult(
                    flat_position=request.flat_position + decision.position,
                    pieces=decision.pieces,
                )
                for decision in decisions
            ]
        }

    def collect(self, state: SourceState) -> dict[str, list[SourcePage]]:
        """Collect split results while preserving page and block order."""
        replacements = {
            result.flat_position: result.pieces
            for result in state.split_results
            if result.pieces is not None
        }
        split_pages: list[SourcePage] = []
        flat_position = 0
        for page in state.image_described_pages:
            split_blocks: list[SourceBlock] = []
            for source_block in page.blocks:
                pieces = replacements.get(flat_position)
                if pieces is None:
                    split_blocks.append(source_block)
                else:
                    split_blocks.extend(
                        SourceBlock(
                            block_type=source_block.block_type,
                            content=piece.content,
                        )
                        for piece in pieces
                    )
                flat_position += 1
            split_pages.append(page.model_copy(update={'blocks': split_blocks}))

        return {'split_pages': split_pages}
