"""LangGraph node for routed source exercise splitting."""

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
from kms.source.models.source import BlockType, SourceBlock
from kms.source.processing.exercise_splitter.inference import (
    ExerciseSplitRouterModule,
    ExerciseSplitterModule,
)
from kms.source.processing.exercise_splitter.models import (
    SplitCandidate,
    SplitRequest,
    SplitResult,
)
from kms.source.processing.splitting import apply_block_splits
from kms.source.processing.state import SourceProcessingState


class ExerciseSplitterWorkerState(TypedDict):
    """State supplied to one dispatched splitter worker."""

    split_request: SplitRequest


class ExerciseSplitterNode:
    """Dispatch, route, split, and collect source blocks."""

    def __init__(
        self,
        router: ExerciseSplitRouterModule,
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
        window: SourceBlockWindow, candidate_positions: set[int]
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
        eligible = {
            position
            for position, block in enumerate(window.target)
            if block.block_type is not BlockType.IMAGE
            and block.image_path is None
        }
        inputs = cls._splitter_inputs(window, eligible)
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
        windows: list[SourceBlockWindow] = []
        flat_position = 0
        while flat_position < len(flat_blocks):
            target_positions: list[int] = []
            target_cost = 0
            best_window = None
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
                candidate_window = select_window(
                    flat_blocks,
                    [*target_positions, position],
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
        blocks = state.image_filtered_blocks
        if not blocks:
            return 'exercise_splitter_collect'
        windows = await run_blocking_work(partial(self._select_windows, blocks))
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
        self, state: ExerciseSplitterWorkerState
    ) -> dict[str, list[SplitResult]]:
        request = state['split_request']
        window = request.window
        eligible = {
            position
            for position, block in enumerate(window.target)
            if block.block_type is not BlockType.IMAGE
            and block.image_path is None
        }
        if not eligible:
            return {
                'exercise_split_results': [
                    SplitResult(flat_position=request.flat_position)
                ]
            }
        decisions = await self._router.acall(**self._router_inputs(window))
        candidates = {
            decision.position
            for decision in decisions
            if decision.requires_split
        }
        if not candidates:
            return {
                'exercise_split_results': [
                    SplitResult(flat_position=request.flat_position)
                ]
            }
        splits = await self._exercise_splitter.acall(
            **self._splitter_inputs(window, candidates)
        )
        return {
            'exercise_split_results': [
                SplitResult(
                    flat_position=request.flat_position + decision.position,
                    pieces=decision.pieces,
                )
                for decision in splits
            ]
        }

    def collect(
        self, state: SourceProcessingState
    ) -> dict[str, list[SourceBlock]]:
        replacements = {
            result.flat_position: [piece.content for piece in result.pieces]
            for result in state.exercise_split_results
            if result.pieces is not None
        }
        return {
            'exercise_split_blocks': apply_block_splits(
                state.image_filtered_blocks, replacements
            )
        }
