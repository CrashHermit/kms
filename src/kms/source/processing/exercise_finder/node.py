"""LangGraph node for discovering ordered exercise units."""

from kms.config.inference import ContextWindowSettings
from kms.core.tokenization.budget import TokenBudget
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.inference.batching import pack_content_batches
from kms.media.budget import ImagePayloadBudget
from kms.source.context import (
    SourceBlockWindow,
    count_source_tokens,
    project_source_block,
    project_source_window,
    select_window,
)
from kms.source.models.source import SourceBlock
from kms.source.models.unit import SourceUnit, SourceUnitKind
from kms.source.processing.exercise_finder.inference import (
    ExerciseBoundaryRouterModule,
    ExerciseStartRouterModule,
)
from kms.source.processing.exercise_finder.models import (
    ExerciseClassificationInput,
)
from kms.source.processing.exercise_finder.settings import (
    ExerciseFinderSettings,
)
from kms.source.processing.state import SourceProcessingState


class ExerciseFinderNode:
    """Find exercise tasks without consuming reserved support blocks."""

    def __init__(
        self,
        start_router: ExerciseStartRouterModule,
        boundary_router: ExerciseBoundaryRouterModule,
        settings: ExerciseFinderSettings,
        *,
        start_budget: TokenBudget,
        boundary_budget: TokenBudget,
        start_token_counters: tuple[MaterialTokenEstimator, ...],
        boundary_token_counters: tuple[MaterialTokenEstimator, ...],
        image_budget: ImagePayloadBudget,
    ) -> None:
        self._start_router, self._boundary_router = (
            start_router,
            boundary_router,
        )
        self._settings, self._start_budget, self._boundary_budget = (
            settings,
            start_budget,
            boundary_budget,
        )
        self._start_token_counters, self._boundary_token_counters = (
            start_token_counters,
            boundary_token_counters,
        )
        self._image_budget = image_budget

    async def run(
        self, state: SourceProcessingState
    ) -> dict[str, list[SourceUnit]]:
        blocks = state.procedure_split_blocks
        reserved = {
            uuid
            for group in state.exercise_support_groups
            for uuid in group.block_uuids
        }
        eligible = [
            position
            for position, block in enumerate(blocks)
            if block.uuid not in reserved
        ]
        starts_costs = count_source_tokens(blocks, self._start_token_counters)
        boundaries_costs = count_source_tokens(
            blocks, self._boundary_token_counters
        )
        units: list[SourceUnit] = []
        for run in self._runs(eligible):
            cursor = 0
            while cursor < len(run):
                stop = min(cursor + self._settings.start_batch_size, len(run))
                targets = run[cursor:stop]
                windows = {
                    position: self._window(
                        blocks,
                        starts_costs,
                        position,
                        self._settings.start_context_window,
                    )
                    for position in targets
                }

                def project_starts(
                    packet: list[int], windows=windows
                ) -> dict[str, object]:
                    return {
                        'items': [
                            self._item(local, windows[position])
                            for local, position in enumerate(packet)
                        ]
                    }

                packet = (
                    await pack_content_batches(
                        [targets],
                        project=project_starts,
                        budget=self._start_budget,
                        image_budget=self._image_budget,
                    )
                )[0]
                decisions = await self._start_router.acall(
                    **project_starts(packet)
                )
                matches = [
                    packet[item.position]
                    for item in decisions
                    if item.is_exercise_start
                ]
                if not matches:
                    cursor += len(packet)
                    continue
                start = min(matches)
                anchor = blocks[start]
                candidate = run.index(start) + 1
                run_end = len(run)
                while candidate < run_end:
                    candidate_targets = run[
                        candidate : min(
                            candidate + self._settings.boundary_batch_size,
                            run_end,
                        )
                    ]
                    boundary_windows = {
                        position: self._window(
                            blocks,
                            boundaries_costs,
                            position,
                            self._settings.boundary_context_window,
                        )
                        for position in candidate_targets
                    }

                    def project_boundaries(
                        values: list[int],
                        anchor=anchor,
                        boundary_windows=boundary_windows,
                    ) -> dict[str, object]:
                        return {
                            'start_block': project_source_block(anchor),
                            'items': [
                                self._item(local, boundary_windows[position])
                                for local, position in enumerate(values)
                            ],
                        }

                    packet = (
                        await pack_content_batches(
                            [candidate_targets],
                            project=project_boundaries,
                            budget=self._boundary_budget,
                            image_budget=self._image_budget,
                        )
                    )[0]
                    decisions = await self._boundary_router.acall(
                        **project_boundaries(packet)
                    )
                    boundaries = [
                        packet[item.position]
                        for item in decisions
                        if item.is_exercise_boundary
                    ]
                    if boundaries:
                        boundary = min(boundaries)
                        units.append(
                            SourceUnit(
                                kind=SourceUnitKind.EXERCISE,
                                blocks=blocks[start:boundary],
                            )
                        )
                        cursor = run.index(boundary)
                        break
                    candidate += len(packet)
                else:
                    boundary = run[-1] + 1
                    units.append(
                        SourceUnit(
                            kind=SourceUnitKind.EXERCISE,
                            blocks=blocks[start:boundary],
                        )
                    )
                    cursor = len(run)
        return {'exercise_units': units}

    @staticmethod
    def _runs(positions: list[int]) -> list[list[int]]:
        runs: list[list[int]] = []
        for position in positions:
            if not runs or position != runs[-1][-1] + 1:
                runs.append([position])
            else:
                runs[-1].append(position)
        return runs

    def _window(
        self,
        blocks: list[SourceBlock],
        costs: list[int],
        position: int,
        settings: ContextWindowSettings,
    ) -> SourceBlockWindow:
        return select_window(
            blocks,
            [position],
            token_counts=costs,
            backward_budget=settings.backward_budget,
            forward_budget=settings.forward_budget,
            target_budget=settings.target_budget,
        )

    @staticmethod
    def _item(
        position: int, window: SourceBlockWindow
    ) -> ExerciseClassificationInput:
        projected = project_source_window(window)
        return ExerciseClassificationInput(
            position=position,
            context_before=projected.context_before,
            target_block=projected.target[0],
            context_after=projected.context_after,
        )
