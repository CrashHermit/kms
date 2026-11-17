"""LangGraph node for discovering complete procedure units."""

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
from kms.source.processing.procedure_finder.inference import (
    ProcedureBoundaryRouterModule,
    ProcedureStartRouterModule,
)
from kms.source.processing.procedure_finder.models import (
    ProcedureClassificationInput,
)
from kms.source.processing.procedure_finder.settings import (
    ProcedureFinderSettings,
)
from kms.source.processing.state import SourceProcessingState


class ProcedureFinderNode:
    def __init__(
        self,
        start_router: ProcedureStartRouterModule,
        boundary_router: ProcedureBoundaryRouterModule,
        settings: ProcedureFinderSettings,
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
        excluded = {
            block.uuid for unit in state.exercise_units for block in unit.blocks
        }
        excluded.update(
            uuid
            for group in state.exercise_support_groups
            for uuid in group.block_uuids
        )
        eligible = [
            position
            for position, block in enumerate(blocks)
            if block.uuid not in excluded
        ]
        starts_costs = count_source_tokens(blocks, self._start_token_counters)
        boundaries_costs = count_source_tokens(
            blocks, self._boundary_token_counters
        )
        units: list[SourceUnit] = []
        for run in self._runs(eligible):
            cursor = 0
            while cursor < len(run):
                targets = run[
                    cursor : min(
                        cursor + self._settings.start_batch_size, len(run)
                    )
                ]
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
                    if item.is_procedure_start
                ]
                if not matches:
                    cursor += len(packet)
                    continue
                start = min(matches)
                anchor = blocks[start]
                candidate = run.index(start) + 1
                while candidate < len(run):
                    target_positions = run[
                        candidate : min(
                            candidate + self._settings.boundary_batch_size,
                            len(run),
                        )
                    ]
                    boundary_windows = {
                        position: self._window(
                            blocks,
                            boundaries_costs,
                            position,
                            self._settings.boundary_context_window,
                        )
                        for position in target_positions
                    }

                    def project_boundaries(
                        packet_positions: list[int],
                        anchor=anchor,
                        boundary_windows=boundary_windows,
                    ) -> dict[str, object]:
                        return {
                            'start_block': project_source_block(anchor),
                            'items': [
                                self._item(local, boundary_windows[position])
                                for local, position in enumerate(
                                    packet_positions
                                )
                            ],
                        }

                    packet = (
                        await pack_content_batches(
                            [target_positions],
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
                        if item.is_procedure_boundary
                    ]
                    if boundaries:
                        boundary = min(boundaries)
                        units.append(
                            SourceUnit(
                                kind=SourceUnitKind.PROCEDURE,
                                blocks=blocks[start:boundary],
                            )
                        )
                        cursor = run.index(boundary)
                        break
                    candidate += len(packet)
                else:
                    end = run[-1] + 1
                    units.append(
                        SourceUnit(
                            kind=SourceUnitKind.PROCEDURE,
                            blocks=blocks[start:end],
                        )
                    )
                    cursor = len(run)
        return {'procedure_units': units}

    @staticmethod
    def _runs(positions: list[int]) -> list[list[int]]:
        runs: list[list[int]] = []
        for position in positions:
            if not runs or position != runs[-1][-1] + 1:
                runs.append([position])
            else:
                runs[-1].append(position)
        return runs

    @staticmethod
    def _window(
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
    ) -> ProcedureClassificationInput:
        projected = project_source_window(window)
        return ProcedureClassificationInput(
            position=position,
            context_before=projected.context_before,
            target_block=projected.target[0],
            context_after=projected.context_after,
        )
