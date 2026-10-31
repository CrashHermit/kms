"""LangGraph node for routed source-block exercise splitting."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.config.inference import ContextWindowSettings
from kms2.core.model.block import SourceBlock
from kms2.core.model.context import PositionedSourceBlock
from kms2.core.model.page import SourcePage
from kms2.core.model.source_pedagogical.exercise_splitter import (
    SplitCandidate,
    SplitRequest,
    SplitResult,
)
from kms2.core.windowing import (
    TextTokenCounter,
    count_text_tokens,
    select_window,
)
from kms2.langgraph.source_pedagogical.state import SourcePedagogicalState
from kms2.module.source_pedagogical.exercise_splitter import (
    ExerciseSplitterModule,
)
from kms2.module.source_pedagogical.exercise_strip_router import (
    ExerciseStripRouterModule,
)


class ExerciseSplitterWorkerState(TypedDict):
    """State supplied to one dispatched exercise splitter worker."""

    split_request: SplitRequest


class ExerciseSplitterNode:
    """Dispatch, route, split, and collect source blocks."""

    def __init__(
        self,
        router: ExerciseStripRouterModule,
        exercise_splitter: ExerciseSplitterModule,
        context_window: ContextWindowSettings,
        *,
        batch_size: int,
        token_counters: tuple[TextTokenCounter, ...],
    ) -> None:
        self._router = router
        self._exercise_splitter = exercise_splitter
        self._context_window = context_window
        self._batch_size = batch_size
        self._token_counters = token_counters

    async def dispatch(
        self, state: SourcePedagogicalState
    ) -> list[Send] | Literal['exercise_splitter_collect']:
        """Dispatch capped context-window requests over source-block batches."""
        flat_blocks = [
            source_block
            for page in state.source_pages
            for source_block in page.blocks
        ]
        token_counts = count_text_tokens(
            [block.content for block in flat_blocks], self._token_counters
        )
        sends: list[Send] = []
        flat_position = 0
        while flat_position < len(flat_blocks):
            window = select_window(
                flat_blocks,
                [flat_position],
                token_counts=token_counts,
                backward_budget=self._context_window.backward_budget,
                forward_budget=self._context_window.forward_budget,
                target_budget=self._context_window.target_budget,
                max_targets=self._batch_size,
            )
            sends.append(
                Send(
                    'exercise_splitter_worker',
                    {
                        'split_request': SplitRequest(
                            flat_position=flat_position,
                            window=window,
                        )
                    },
                )
            )
            flat_position += len(window.target)
        return sends or 'exercise_splitter_collect'

    async def worker(
        self,
        state: ExerciseSplitterWorkerState,
    ) -> dict[str, list[SplitResult]]:
        """Route and split one dispatched source-block window."""
        request = state['split_request']
        window = request.window
        decisions = await self._router.acall(
            context_before=window.context_before,
            targets=[
                PositionedSourceBlock(
                    position=position, source_block=source_block
                )
                for position, source_block in enumerate(window.target)
            ],
            context_after=window.context_after,
        )
        candidates = [
            SplitCandidate(
                position=decision.position,
                source_block=window.target[decision.position],
            )
            for decision in sorted(
                decisions, key=lambda decision: decision.position
            )
            if decision.contains_multiple_exercises
        ]
        if not candidates:
            return {
                'split_results': [
                    SplitResult(flat_position=request.flat_position)
                ]
            }

        decisions = await self._exercise_splitter.acall(
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

    def collect(
        self, state: SourcePedagogicalState
    ) -> dict[str, list[SourcePage]]:
        """Collect split results while preserving page and block order."""
        replacements = {
            result.flat_position: result.pieces
            for result in state.split_results
            if result.pieces is not None
        }
        split_pages: list[SourcePage] = []
        flat_position = 0
        for page in state.source_pages:
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
