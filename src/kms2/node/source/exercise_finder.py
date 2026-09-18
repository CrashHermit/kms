"""LangGraph node for discovering exercise components."""

from kms2.config.source import ExerciseFinderSettings
from kms2.core.model import ExerciseComponent
from kms2.core.windowing import select_window
from kms2.langgraph.source.state import SourceState
from kms2.module.source.exercise_finder import (
    ExerciseBoundaryRouterModule,
    ExerciseStartRouterModule,
)


class ExerciseFinderNode:
    """Find ordered exercise components without mutating source blocks."""

    def __init__(
        self,
        start_router: ExerciseStartRouterModule,
        boundary_router: ExerciseBoundaryRouterModule,
        settings: ExerciseFinderSettings,
    ) -> None:
        self._start_router = start_router
        self._boundary_router = boundary_router
        self._settings = settings

    async def run(
        self, state: SourceState
    ) -> dict[str, list[ExerciseComponent]]:
        """Discover exercise components across the unclaimed source stream."""
        blocks = [block for page in state.split_pages for block in page.blocks]
        instruction_members = {
            block_uuid
            for instruction in state.instructions
            for block_uuid in instruction.member_block_uuids
        }
        eligible_positions = [
            position
            for position, block in enumerate(blocks)
            if block.uuid not in instruction_members
        ]
        eligible_blocks = [blocks[position] for position in eligible_positions]
        components: list[ExerciseComponent] = []
        cursor = 0

        while cursor < len(eligible_blocks):
            start_window = select_window(
                eligible_blocks,
                [cursor],
                backward_budget=self._settings.start_context_window.backward_budget,
                forward_budget=self._settings.start_context_window.forward_budget,
                target_budget=self._settings.start_context_window.target_budget,
            )
            is_start = await self._start_router.aforward(
                context_before=start_window.context_before,
                target_block=start_window.target[0],
                context_after=start_window.context_after,
            )
            if not is_start:
                cursor += 1
                continue

            start_block = start_window.target[0]
            candidate = cursor + 1
            while candidate < len(eligible_blocks):
                boundary_window = select_window(
                    eligible_blocks,
                    [candidate],
                    backward_budget=self._settings.boundary_context_window.backward_budget,
                    forward_budget=self._settings.boundary_context_window.forward_budget,
                    target_budget=self._settings.boundary_context_window.target_budget,
                )
                is_boundary = await self._boundary_router.aforward(
                    start_block=start_block,
                    context_before=boundary_window.context_before,
                    candidate_block=boundary_window.target[0],
                    context_after=boundary_window.context_after,
                )
                if is_boundary:
                    components.append(
                        ExerciseComponent(
                            member_block_uuids=[
                                eligible_blocks[position].uuid
                                for position in range(cursor, candidate)
                            ]
                        )
                    )
                    cursor = candidate
                    break
                candidate += 1
            else:
                components.append(
                    ExerciseComponent(
                        member_block_uuids=[
                            eligible_blocks[position].uuid
                            for position in range(cursor, len(eligible_blocks))
                        ]
                    )
                )
                cursor = len(eligible_blocks)

        return {'exercise_components': components}
