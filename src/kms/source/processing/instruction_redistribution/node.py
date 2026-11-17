"""Detect shared exercise support and redistribute applicable copies."""

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
from kms.source.processing.instruction_redistribution.inference import (
    InstructionApplicabilityModule,
    InstructionBoundaryRouterModule,
    InstructionStartRouterModule,
)
from kms.source.processing.instruction_redistribution.models import (
    ExerciseSupportGroup,
    InstructionApplicabilityInput,
    InstructionClassificationInput,
)
from kms.source.processing.instruction_redistribution.settings import (
    InstructionRedistributionSettings,
)
from kms.source.processing.state import SourceProcessingState


class InstructionRedistributionNode:
    def __init__(
        self,
        start_router: InstructionStartRouterModule,
        boundary_router: InstructionBoundaryRouterModule,
        applicability: InstructionApplicabilityModule,
        settings: InstructionRedistributionSettings,
        *,
        start_budget: TokenBudget,
        boundary_budget: TokenBudget,
        applicability_budget: TokenBudget,
        start_token_counters: tuple[MaterialTokenEstimator, ...],
        boundary_token_counters: tuple[MaterialTokenEstimator, ...],
        applicability_token_counters: tuple[MaterialTokenEstimator, ...],
        image_budget: ImagePayloadBudget,
        applicability_image_budget: ImagePayloadBudget,
    ) -> None:
        self._start_router, self._boundary_router, self._applicability = (
            start_router,
            boundary_router,
            applicability,
        )
        self._settings = settings
        (
            self._start_budget,
            self._boundary_budget,
            self._applicability_budget,
        ) = start_budget, boundary_budget, applicability_budget
        self._start_token_counters, self._boundary_token_counters = (
            start_token_counters,
            boundary_token_counters,
        )
        self._applicability_token_counters = applicability_token_counters
        self._image_budget, self._applicability_image_budget = (
            image_budget,
            applicability_image_budget,
        )

    async def detect(
        self, state: SourceProcessingState
    ) -> dict[str, list[ExerciseSupportGroup]]:
        blocks = state.procedure_split_blocks
        starts_cost = count_source_tokens(blocks, self._start_token_counters)
        boundaries_cost = count_source_tokens(
            blocks, self._boundary_token_counters
        )
        groups, cursor = [], 0
        while cursor < len(blocks):
            stop = min(
                cursor + self._settings.detection.start_batch_size, len(blocks)
            )
            windows = {
                i: self._window(
                    blocks,
                    starts_cost,
                    i,
                    self._settings.detection.start_context_window,
                )
                for i in range(cursor, stop)
            }

            def project_starts(
                positions: list[int], windows=windows
            ) -> dict[str, object]:
                return {
                    'items': [
                        self._classification(local, windows[position])
                        for local, position in enumerate(positions)
                    ]
                }

            batch = (
                await pack_content_batches(
                    [list(windows)],
                    project=project_starts,
                    budget=self._start_budget,
                    image_budget=self._image_budget,
                )
            )[0]
            decisions = await self._start_router.acall(**project_starts(batch))
            matches = [
                batch[item.position]
                for item in decisions
                if item.is_instruction_start
            ]
            if not matches:
                cursor = batch[-1] + 1
                continue
            start = min(matches)
            candidate = start + 1
            while candidate < len(blocks):
                stop = min(
                    candidate + self._settings.detection.boundary_batch_size,
                    len(blocks),
                )
                windows = {
                    i: self._window(
                        blocks,
                        boundaries_cost,
                        i,
                        self._settings.detection.boundary_context_window,
                    )
                    for i in range(candidate, stop)
                }

                def project_boundaries(
                    positions: list[int], start=start, windows=windows
                ) -> dict[str, object]:
                    return {
                        'start_block': project_source_block(blocks[start]),
                        'items': [
                            self._classification(local, windows[position])
                            for local, position in enumerate(positions)
                        ],
                    }

                batch = (
                    await pack_content_batches(
                        [list(windows)],
                        project=project_boundaries,
                        budget=self._boundary_budget,
                        image_budget=self._image_budget,
                    )
                )[0]
                decisions = await self._boundary_router.acall(
                    **project_boundaries(batch)
                )
                bounds = [
                    batch[item.position]
                    for item in decisions
                    if item.is_instruction_boundary
                ]
                if bounds:
                    end = min(bounds)
                    groups.append(
                        ExerciseSupportGroup(
                            block_uuids=[
                                block.uuid for block in blocks[start:end]
                            ]
                        )
                    )
                    cursor = end
                    break
                candidate = batch[-1] + 1
            else:
                groups.append(
                    ExerciseSupportGroup(
                        block_uuids=[block.uuid for block in blocks[start:]]
                    )
                )
                cursor = len(blocks)
        return {'exercise_support_groups': groups}

    async def run(
        self, state: SourceProcessingState
    ) -> dict[str, list[SourceUnit]]:
        blocks = state.procedure_split_blocks
        positions = {block.uuid: index for index, block in enumerate(blocks)}
        exercises = [
            (index, unit)
            for index, unit in enumerate(state.units)
            if unit.kind is SourceUnitKind.EXERCISE
        ]
        copies: dict[str, list[tuple[int, SourceBlock]]] = {}
        incorporated: set[str] = set()
        costs = (
            count_source_tokens(blocks, self._applicability_token_counters)
            if exercises
            else []
        )
        for group in state.exercise_support_groups:
            support = [blocks[positions[uuid]] for uuid in group.block_uuids]
            projected_support = [
                project_source_block(block) for block in support
            ]
            windows = {
                index: select_window(
                    blocks,
                    [positions[block.uuid] for block in unit.blocks],
                    token_counts=costs,
                    backward_budget=self._settings.applicability.context_window.backward_budget,
                    forward_budget=self._settings.applicability.context_window.forward_budget,
                )
                for index, unit in exercises
            }
            exercise_by_index = dict(exercises)
            exercise_indices = list(exercise_by_index)

            def project_items(
                packet: list[int],
                projected_support=projected_support,
                windows=windows,
                exercise_by_index=exercise_by_index,
            ) -> dict[str, object]:
                return {
                    'instruction_blocks': projected_support,
                    'items': [
                        InstructionApplicabilityInput(
                            position=local,
                            context_before=(
                                project_source_window(
                                    windows[index]
                                ).context_before
                            ),
                            exercise_blocks=[
                                project_source_block(block)
                                for block in exercise_by_index[index].blocks
                            ],
                            context_after=project_source_window(
                                windows[index]
                            ).context_after,
                        )
                        for local, index in enumerate(packet)
                    ],
                }

            incoming = [
                exercise_indices[
                    start : start + self._settings.applicability.batch_size
                ]
                for start in range(
                    0,
                    len(exercise_indices),
                    self._settings.applicability.batch_size,
                )
            ]
            batches = (
                await pack_content_batches(
                    incoming,
                    project=project_items,
                    budget=self._applicability_budget,
                    image_budget=self._applicability_image_budget,
                )
                if exercises
                else []
            )
            for packet in batches:
                decisions = await self._applicability.acall(
                    **project_items(packet)
                )
                for decision in decisions:
                    if decision.applies:
                        unit = exercise_by_index[packet[decision.position]]
                        copies.setdefault(unit.uuid, []).extend(
                            (positions[block.uuid], self._copy_block(block))
                            for block in support
                        )
                        incorporated.update(group.block_uuids)
        updated = []
        for unit in state.units:
            if unit.uuid in copies:
                members = [
                    (positions[block.uuid], block) for block in unit.blocks
                ] + copies[unit.uuid]
                updated.append(
                    unit.model_copy(
                        update={
                            'blocks': [
                                block
                                for _, block in sorted(
                                    members, key=lambda pair: pair[0]
                                )
                            ]
                        }
                    )
                )
            elif unit.kind is SourceUnitKind.CONTENT and all(
                block.uuid in incorporated for block in unit.blocks
            ):
                continue
            else:
                updated.append(unit)
        return {'units': updated}

    @staticmethod
    def _copy_block(block: SourceBlock) -> SourceBlock:
        return SourceBlock(
            block_type=block.block_type,
            content=block.content,
            crop_path=block.crop_path,
            crop_sha256=block.crop_sha256,
            crop_bbox=block.crop_bbox,
            image_path=block.image_path,
            image_sha256=block.image_sha256,
        )

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
    def _classification(
        position: int, window: SourceBlockWindow
    ) -> InstructionClassificationInput:
        projected = project_source_window(window)
        return InstructionClassificationInput(
            position=position,
            context_before=projected.context_before,
            target_block=projected.target[0],
            context_after=projected.context_after,
        )
