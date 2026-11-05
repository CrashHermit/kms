"""LangGraph node for routed source-block exercise splitting."""

from typing import Literal, TypedDict

import dspy
from langgraph.types import Send

from kms.config.inference import ContextWindowSettings
from kms.core.content import MultimodalContentBuilder
from kms.core.media_budget import ImagePayloadBudget
from kms.core.model.block import SourceBlock
from kms.core.model.block_types import BlockType
from kms.core.model.context import PositionedSourceBlock
from kms.core.model.page import SourcePage
from kms.core.model.source_pedagogical.exercise_splitter import (
    SplitCandidate,
    SplitRequest,
    SplitResult,
)
from kms.core.source_context import count_source_tokens, project_source_window
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.core.windowing import select_window
from kms.langgraph.source_pedagogical.state import SourcePedagogicalState
from kms.module.source_pedagogical.exercise_splitter import (
    ExerciseSplitterModule,
)
from kms.module.source_pedagogical.exercise_strip_router import (
    ExerciseStripRouterModule,
)
from kms.service.content import load_content_images


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
    ) -> None:
        self._router = router
        self._exercise_splitter = exercise_splitter
        self._context_window = context_window
        self._batch_size = batch_size
        self._token_counters = token_counters
        self._image_budget = image_budget

    async def dispatch(
        self, state: SourcePedagogicalState
    ) -> list[Send] | Literal['exercise_splitter_collect']:
        """Dispatch capped context-window requests over source-block batches."""
        flat_blocks = [
            source_block
            for page in state.source_pages
            for source_block in page.blocks
        ]
        token_counts = count_source_tokens(flat_blocks, self._token_counters)
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
        """Route and split one indivisible canonical source window."""
        request = state['split_request']
        window = request.window
        eligible_positions = {
            position
            for position, block in enumerate(window.target)
            if block.block_type != BlockType.IMAGE and not block.assets
        }
        if not eligible_positions:
            return {
                'split_results': [
                    SplitResult(flat_position=request.flat_position)
                ]
            }

        router_builder = MultimodalContentBuilder()
        router_window = project_source_window(window, router_builder)
        router_images = await load_content_images(
            router_builder.images, budget=self._image_budget
        )
        decisions = await self._router.acall(
            context_before=router_window.context_before,
            targets=[
                PositionedSourceBlock(position=position, source_block=block)
                for position, block in enumerate(router_window.target)
            ],
            context_after=router_window.context_after,
            content_images=[dspy.Image(payload) for payload in router_images],
        )
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

        splitter_builder = MultimodalContentBuilder()
        splitter_window = project_source_window(window, splitter_builder)
        reference_blocks = [
            PositionedSourceBlock(
                position=position,
                source_block=splitter_window.target[position],
            )
            for position in range(len(window.target))
            if position not in candidate_positions
        ]
        candidates = [
            SplitCandidate(
                position=position,
                source_block=splitter_window.target[position],
            )
            for position in candidate_positions
        ]
        splitter_images = await load_content_images(
            splitter_builder.images, budget=self._image_budget
        )
        splits = await self._exercise_splitter.acall(
            context_before=splitter_window.context_before,
            candidates=candidates,
            reference_blocks=reference_blocks,
            context_after=splitter_window.context_after,
            content_images=[dspy.Image(payload) for payload in splitter_images],
        )
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
                            origin_uuid=source_block.uuid,
                        )
                        for piece in pieces
                    )
                flat_position += 1
            split_pages.append(page.model_copy(update={'blocks': split_blocks}))

        return {'split_pages': split_pages}
