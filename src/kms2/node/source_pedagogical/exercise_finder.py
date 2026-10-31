"""LangGraph node for discovering exercise components."""

import json

from kms2.config.inference import ContextWindowSettings
from kms2.config.source_pedagogical import ExerciseFinderSettings
from kms2.core.model.block import SourceBlock
from kms2.core.model.source_pedagogical.pedagogical import (
    ExerciseBoundaryDecision,
    ExerciseClassificationInput,
    ExerciseComponent,
    ExerciseStartDecision,
)
from kms2.core.windowing import (
    TextTokenCounter,
    TokenBudget,
    count_text_tokens,
    pack_items,
    select_window,
)
from kms2.langgraph.source_pedagogical.state import SourcePedagogicalState
from kms2.module.source_pedagogical.exercise_finder import (
    ExerciseBoundaryRouterModule,
    ExerciseStartRouterModule,
)


def _serialized_item_token_count(
    item: ExerciseClassificationInput, counter: TextTokenCounter
) -> int:
    """Measure a complete item's serialized request cost."""
    serialized = json.dumps(
        {'items': [item.model_dump(mode='json')]},
        ensure_ascii=False,
        separators=(',', ':'),
    )
    return counter.count_texts([serialized])[0]


class ExerciseFinderNode:
    """Find ordered exercise components without mutating source blocks."""

    def __init__(
        self,
        start_router: ExerciseStartRouterModule,
        boundary_router: ExerciseBoundaryRouterModule,
        settings: ExerciseFinderSettings,
        *,
        start_budget: TokenBudget,
        boundary_budget: TokenBudget,
        start_token_counters: tuple[TextTokenCounter, ...],
        boundary_token_counters: tuple[TextTokenCounter, ...],
    ) -> None:
        self._start_router = start_router
        self._boundary_router = boundary_router
        self._settings = settings
        self._start_budget = start_budget
        self._boundary_budget = boundary_budget
        self._start_token_counters = start_token_counters
        self._boundary_token_counters = boundary_token_counters

    async def run(
        self, state: SourcePedagogicalState
    ) -> dict[str, list[ExerciseComponent]]:
        """Discover exercise components across the unclaimed source stream."""
        blocks = [block for page in state.split_pages for block in page.blocks]
        instruction_members = {
            block_uuid
            for instruction in state.instructions
            for block_uuid in instruction.member_block_uuids
        }
        eligible_blocks = [
            block for block in blocks if block.uuid not in instruction_members
        ]
        start_token_counts = count_text_tokens(
            [block.content for block in eligible_blocks],
            self._start_token_counters,
        )
        boundary_token_counts = count_text_tokens(
            [block.content for block in eligible_blocks],
            self._boundary_token_counters,
        )
        components: list[ExerciseComponent] = []
        cursor = 0

        while cursor < len(eligible_blocks):
            start_candidates = self._classification_items(
                eligible_blocks,
                start_token_counts,
                cursor,
                min(
                    cursor + self._settings.start_batch_size,
                    len(eligible_blocks),
                ),
                self._settings.start_context_window,
            )
            batch = pack_items(
                start_candidates,
                token_counts=[
                    _serialized_item_token_count(
                        item, self._start_budget.counter
                    )
                    for item in start_candidates
                ],
                token_budget=self._start_budget.token_limit,
                max_items=self._settings.start_batch_size,
            )[0]
            decisions: list[
                ExerciseStartDecision
            ] = await self._start_router.acall(items=batch)
            starts = [
                cursor + decision.position
                for decision in decisions
                if decision.is_exercise_start
            ]
            if not starts:
                cursor += len(batch)
                continue

            start_position = min(starts)
            start_block = batch[start_position - cursor].target_block
            candidate = start_position + 1
            while candidate < len(eligible_blocks):
                boundary_candidates = self._classification_items(
                    eligible_blocks,
                    boundary_token_counts,
                    candidate,
                    min(
                        candidate + self._settings.boundary_batch_size,
                        len(eligible_blocks),
                    ),
                    self._settings.boundary_context_window,
                )
                anchor_serialized = json.dumps(
                    {'start_block': start_block.model_dump(mode='json')},
                    ensure_ascii=False,
                    separators=(',', ':'),
                )
                anchor_count = self._boundary_budget.counter.count_texts(
                    [anchor_serialized]
                )[0]
                batch = pack_items(
                    boundary_candidates,
                    token_counts=[
                        _serialized_item_token_count(
                            item, self._boundary_budget.counter
                        )
                        for item in boundary_candidates
                    ],
                    token_budget=self._boundary_budget.token_limit
                    - anchor_count,
                    max_items=self._settings.boundary_batch_size,
                )[0]
                boundary_decisions: list[
                    ExerciseBoundaryDecision
                ] = await self._boundary_router.acall(
                    start_block=start_block,
                    items=batch,
                )
                boundaries = [
                    candidate + decision.position
                    for decision in boundary_decisions
                    if decision.is_exercise_boundary
                ]
                if boundaries:
                    boundary = min(boundaries)
                    components.append(
                        ExerciseComponent(
                            member_block_uuids=[
                                eligible_blocks[position].uuid
                                for position in range(start_position, boundary)
                            ]
                        )
                    )
                    cursor = boundary
                    break
                candidate += len(batch)
            else:
                components.append(
                    ExerciseComponent(
                        member_block_uuids=[
                            eligible_blocks[position].uuid
                            for position in range(
                                start_position, len(eligible_blocks)
                            )
                        ]
                    )
                )
                cursor = len(eligible_blocks)

        return {'exercise_components': components}

    def _classification_items(
        self,
        blocks: list[SourceBlock],
        token_counts: list[int],
        first_position: int,
        stop_position: int,
        context_window: ContextWindowSettings,
    ) -> list[ExerciseClassificationInput]:
        """Build bounded, independently windowed candidates for one batch."""
        items: list[ExerciseClassificationInput] = []
        for position in range(first_position, stop_position):
            window = select_window(
                blocks,
                [position],
                token_counts=token_counts,
                backward_budget=context_window.backward_budget,
                forward_budget=context_window.forward_budget,
                target_budget=context_window.target_budget,
            )
            items.append(
                ExerciseClassificationInput(
                    position=position - first_position,
                    context_before=window.context_before,
                    target_block=window.target[0],
                    context_after=window.context_after,
                )
            )
        return items
