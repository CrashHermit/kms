"""LangGraph node for routed source-block exercise splitting."""

from functools import partial
from typing import Literal, TypedDict

from langgraph.types import Send

from kms.concurrency import run_blocking_work
from kms.config.inference import ContextWindowSettings
from kms.core.tokenization.budget import InputBudgetExceeded, TokenBudget
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.inference.generation import fits_generation_request
from kms.media.budget import ImagePayloadBudget
from kms.source.context import (
    PositionedSourceBlock,
    SourceBlockWindow,
    count_source_tokens,
    project_source_window,
    select_window,
)
from kms.source.models.source import BlockType, SourceBlock, SourcePage
from kms.source.processing.exercise_splitter.inference import (
    ExerciseSplitterModule,
    ExerciseStripRouterModule,
)
from kms.source.processing.exercise_splitter.models import (
    SplitCandidate,
    SplitRequest,
    SplitResult,
)
from kms.source.processing.state import SourceProcessingState


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
        token_counters: tuple[MaterialTokenEstimator, ...],
        image_budget: ImagePayloadBudget,
        router_budget: TokenBudget,
        splitter_budget: TokenBudget,
    ) -> None:
        self._router = router
        self._exercise_splitter = exercise_splitter
        self._context_window = context_window
        self._batch_size = batch_size
        self._token_counters = token_counters
        self._image_budget = image_budget
        self._router_budget = router_budget
        self._splitter_budget = splitter_budget

    @staticmethod
    def _router_inputs(window: SourceBlockWindow) -> dict[str, object]:
        projected = project_source_window(window)
        return {
            'context_before': projected.context_before,
            'targets': [
                PositionedSourceBlock(position=position, source_block=block)
                for position, block in enumerate(projected.target)
            ],
            'context_after': projected.context_after,
        }

    @staticmethod
    def _splitter_inputs(
        window: SourceBlockWindow,
        candidate_positions: set[int],
    ) -> dict[str, object]:
        projected = project_source_window(window)
        return {
            'context_before': projected.context_before,
            'candidates': [
                SplitCandidate(
                    position=position, source_block=projected.target[position]
                )
                for position in sorted(candidate_positions)
            ],
            'reference_blocks': [
                PositionedSourceBlock(position=position, source_block=block)
                for position, block in enumerate(projected.target)
                if position not in candidate_positions
            ],
            'context_after': projected.context_after,
        }

    @classmethod
    def _splitter_capacity_inputs(
        cls, window: SourceBlockWindow
    ) -> dict[str, object]:
        eligible_positions = {
            position
            for position, block in enumerate(window.target)
            if block.block_type != BlockType.IMAGE and block.image_path is None
        }
        inputs = cls._splitter_inputs(window, eligible_positions)
        projected = project_source_window(window)
        inputs['reference_blocks'] = [
            PositionedSourceBlock(position=position, source_block=block)
            for position, block in enumerate(projected.target)
        ]
        return inputs

    def _select_windows(
        self, flat_blocks: list[SourceBlock]
    ) -> list[SourceBlockWindow]:
        token_counts = count_source_tokens(flat_blocks, self._token_counters)
        windows = []
        flat_position = 0
        while flat_position < len(flat_blocks):
            target_positions: list[int] = []
            target_cost = 0
            best_window: SourceBlockWindow | None = None
            for position in range(
                flat_position,
                min(
                    flat_position
                    + (
                        self._batch_size
                        if self._context_window.target_budget > 0
                        else 1
                    ),
                    len(flat_blocks),
                ),
            ):
                cost = token_counts[position]
                if (
                    target_positions
                    and self._context_window.target_budget > 0
                    and target_cost + cost > self._context_window.target_budget
                ):
                    break
                candidate_positions = [*target_positions, position]
                candidate_window = select_window(
                    flat_blocks,
                    candidate_positions,
                    token_counts=token_counts,
                    backward_budget=self._context_window.backward_budget,
                    forward_budget=self._context_window.forward_budget,
                )
                fits = fits_generation_request(
                    self._router_inputs(candidate_window),
                    budget=self._router_budget,
                    image_budget=self._image_budget,
                ) and fits_generation_request(
                    self._splitter_capacity_inputs(candidate_window),
                    budget=self._splitter_budget,
                    image_budget=self._image_budget,
                )
                if not fits:
                    if best_window is None:
                        raise InputBudgetExceeded(
                            'One source block exceeds the configured splitter request capacity'
                        )
                    break
                target_positions.append(position)
                target_cost += cost
                best_window = candidate_window
            assert best_window is not None
            windows.append(best_window)
            flat_position += len(best_window.target)
        return windows

    async def dispatch(
        self, state: SourceProcessingState
    ) -> list[Send] | Literal['exercise_splitter_collect']:
        """Dispatch largest contiguous windows fitting both complete tasks."""
        flat_blocks = [
            block
            for page in state.image_filtered_pages
            for block in page.blocks
        ]
        if not flat_blocks:
            return 'exercise_splitter_collect'
        windows = await run_blocking_work(
            partial(self._select_windows, flat_blocks)
        )
        flat_position = 0
        sends = []
        for window in windows:
            sends.append(
                Send(
                    'exercise_splitter_worker',
                    {
                        'split_request': SplitRequest(
                            flat_position=flat_position, window=window
                        )
                    },
                )
            )
            flat_position += len(window.target)
        return sends

    async def worker(
        self,
        state: ExerciseSplitterWorkerState,
    ) -> dict[str, list[SplitResult]]:
        """Route and split one indivisible canonical source window."""
        request = state['split_request']
        window = request.window
        eligible_positions = {
            position
            for position, block in enumerate(window.target)
            if block.block_type != BlockType.IMAGE and block.image_path is None
        }
        if not eligible_positions:
            return {
                'split_results': [
                    SplitResult(flat_position=request.flat_position)
                ]
            }

        router_inputs = self._router_inputs(window)
        decisions = await self._router.acall(**router_inputs)
        candidate_positions = sorted(
            decision.position
            for decision in decisions
            if decision.contains_multiple_exercises
            and decision.position in eligible_positions
        )
        if not candidate_positions:
            return {
                'split_results': [
                    SplitResult(flat_position=request.flat_position)
                ]
            }

        splitter_inputs = self._splitter_inputs(
            window, set(candidate_positions)
        )
        splits = await self._exercise_splitter.acall(**splitter_inputs)
        return {
            'split_results': [
                SplitResult(
                    flat_position=request.flat_position + decision.position,
                    pieces=decision.pieces,
                )
                for decision in splits
            ]
        }

    def collect(
        self, state: SourceProcessingState
    ) -> dict[str, list[SourcePage]]:
        """Collect split results while preserving page and block order."""
        replacements = {
            result.flat_position: result.pieces
            for result in state.split_results
            if result.pieces is not None
        }
        split_pages: list[SourcePage] = []
        flat_position = 0
        for page in state.image_filtered_pages:
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
                            origin_uuid=source_block.uuid,
                        )
                        for piece in pieces
                    )
                flat_position += 1
            split_pages.append(page.model_copy(update={'blocks': split_blocks}))

        return {'split_pages': split_pages}
