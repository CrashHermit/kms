"""LangGraph node for routed source procedure splitting."""

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
from kms.source.processing.procedure_splitter.inference import (
    ProcedureSplitRouterModule,
    ProcedureSplitterModule,
)
from kms.source.processing.procedure_splitter.models import (
    ProcedureSplitCandidate,
    ProcedureSplitRequest,
    ProcedureSplitResult,
)
from kms.source.processing.splitting import apply_block_splits
from kms.source.processing.state import SourceProcessingState


class ProcedureSplitterWorkerState(TypedDict):
    procedure_split_request: ProcedureSplitRequest


class ProcedureSplitterNode:
    def __init__(
        self,
        router: ProcedureSplitRouterModule,
        splitter: ProcedureSplitterModule,
        context_window: ContextWindowSettings,
        *,
        batch_size: int,
        token_counters: tuple[MaterialTokenEstimator, ...],
        image_budget: ImagePayloadBudget,
        router_budget: TokenBudget,
        splitter_budget: TokenBudget,
    ) -> None:
        self._router, self._splitter = router, splitter
        self._context_window, self._batch_size = context_window, batch_size
        self._token_counters, self._image_budget = token_counters, image_budget
        self._router_budget, self._splitter_budget = (
            router_budget,
            splitter_budget,
        )

    @staticmethod
    def _router_inputs(window: SourceBlockWindow) -> dict[str, object]:
        projected = project_source_window(window)
        return {
            'context_before': projected.context_before,
            'targets': [
                PositionedSourceBlock(position=i, source_block=block)
                for i, block in enumerate(projected.target)
            ],
            'context_after': projected.context_after,
        }

    @staticmethod
    def _splitter_inputs(
        window: SourceBlockWindow, candidates: set[int]
    ) -> dict[str, object]:
        projected = project_source_window(window)
        return {
            'context_before': projected.context_before,
            'candidates': [
                ProcedureSplitCandidate(
                    position=i, source_block=projected.target[i]
                )
                for i in sorted(candidates)
            ],
            'reference_blocks': [
                PositionedSourceBlock(position=i, source_block=block)
                for i, block in enumerate(projected.target)
                if i not in candidates
            ],
            'context_after': projected.context_after,
        }

    def _select_windows(
        self, blocks: list[SourceBlock]
    ) -> list[SourceBlockWindow]:
        costs = count_source_tokens(blocks, self._token_counters)
        windows, offset = [], 0
        while offset < len(blocks):
            targets, cost, best = [], 0, None
            limit = min(
                offset
                + (
                    self._batch_size
                    if self._context_window.target_budget > 0
                    else 1
                ),
                len(blocks),
            )
            for position in range(offset, limit):
                if (
                    targets
                    and self._context_window.target_budget > 0
                    and cost + costs[position]
                    > self._context_window.target_budget
                ):
                    break
                candidate = select_window(
                    blocks,
                    [*targets, position],
                    token_counts=costs,
                    backward_budget=self._context_window.backward_budget,
                    forward_budget=self._context_window.forward_budget,
                )
                router_fits = fits_generation_request(
                    self._router_inputs(candidate),
                    budget=self._router_budget,
                    image_budget=self._image_budget,
                )
                splitter_inputs = self._splitter_inputs(
                    candidate,
                    {
                        i
                        for i, block in enumerate(candidate.target)
                        if block.block_type is not BlockType.IMAGE
                        and block.image_path is None
                    },
                )
                splitter_fits = fits_generation_request(
                    splitter_inputs,
                    budget=self._splitter_budget,
                    image_budget=self._image_budget,
                )
                if not router_fits or not splitter_fits:
                    if best is None:
                        raise InputBudgetExceeded(
                            'One source block exceeds the configured procedure splitter request capacity'
                        )
                    break
                targets.append(position)
                cost += costs[position]
                best = candidate
            assert best is not None
            windows.append(best)
            offset += len(best.target)
        return windows

    async def dispatch(
        self, state: SourceProcessingState
    ) -> list[Send] | Literal['procedure_splitter_collect']:
        blocks = state.exercise_split_blocks
        if not blocks:
            return 'procedure_splitter_collect'
        windows = await run_blocking_work(partial(self._select_windows, blocks))
        sends, offset = [], 0
        for window in windows:
            sends.append(
                Send(
                    'procedure_splitter_worker',
                    {
                        'procedure_split_request': ProcedureSplitRequest(
                            flat_position=offset, window=window
                        )
                    },
                )
            )
            offset += len(window.target)
        return sends

    async def worker(
        self, state: ProcedureSplitterWorkerState
    ) -> dict[str, list[ProcedureSplitResult]]:
        request = state['procedure_split_request']
        eligible = {
            i
            for i, block in enumerate(request.window.target)
            if block.block_type is not BlockType.IMAGE
            and block.image_path is None
        }
        if not eligible:
            return {
                'procedure_split_results': [
                    ProcedureSplitResult(flat_position=request.flat_position)
                ]
            }
        decisions = await self._router.acall(
            **self._router_inputs(request.window)
        )
        candidates = {
            item.position for item in decisions if item.requires_split
        }
        if not candidates:
            return {
                'procedure_split_results': [
                    ProcedureSplitResult(flat_position=request.flat_position)
                ]
            }
        splits = await self._splitter.acall(
            **self._splitter_inputs(request.window, candidates)
        )
        return {
            'procedure_split_results': [
                ProcedureSplitResult(
                    flat_position=request.flat_position + item.position,
                    pieces=item.pieces,
                )
                for item in splits
            ]
        }

    def collect(
        self, state: SourceProcessingState
    ) -> dict[str, list[SourceBlock]]:
        replacements = {
            item.flat_position: [piece.content for piece in item.pieces]
            for item in state.procedure_split_results
            if item.pieces is not None
        }
        return {
            'procedure_split_blocks': apply_block_splits(
                state.exercise_split_blocks, replacements
            )
        }
