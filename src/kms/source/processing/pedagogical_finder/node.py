"""LangGraph node for discovering pedagogical source components."""

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
from kms.source.models.material import PedagogicalComponent
from kms.source.models.source import SourceBlock
from kms.source.processing.pedagogical_finder.inference import (
    PedagogicalBoundaryRouterModule,
    PedagogicalStartRouterModule,
)
from kms.source.processing.pedagogical_finder.models import (
    PedagogicalClassificationInput,
)
from kms.source.processing.pedagogical_finder.settings import (
    PedagogicalFinderSettings,
)
from kms.source.processing.state import SourceProcessingState


class PedagogicalFinderNode:
    """Find ordered pedagogical components outside claimed source blocks."""

    def __init__(
        self,
        start_router: PedagogicalStartRouterModule,
        boundary_router: PedagogicalBoundaryRouterModule,
        settings: PedagogicalFinderSettings,
        *,
        start_budget: TokenBudget,
        boundary_budget: TokenBudget,
        start_token_counters: tuple[MaterialTokenEstimator, ...],
        boundary_token_counters: tuple[MaterialTokenEstimator, ...],
        image_budget: ImagePayloadBudget,
    ) -> None:
        self._start_router = start_router
        self._boundary_router = boundary_router
        self._settings = settings
        self._start_budget = start_budget
        self._boundary_budget = boundary_budget
        self._start_token_counters = start_token_counters
        self._boundary_token_counters = boundary_token_counters
        self._image_budget = image_budget

    async def run(
        self, state: SourceProcessingState
    ) -> dict[str, list[PedagogicalComponent]]:
        """Discover components without modifying canonical source blocks."""
        blocks = [block for page in state.split_pages for block in page.blocks]
        claimed_members = {
            block_uuid
            for instruction in state.instructions
            for block_uuid in instruction.member_block_uuids
        }
        claimed_members.update(
            block_uuid
            for component in state.exercise_components
            for block_uuid in component.member_block_uuids
        )
        eligible_blocks = [
            block for block in blocks if block.uuid not in claimed_members
        ]
        start_token_counts = count_source_tokens(
            eligible_blocks, self._start_token_counters
        )
        boundary_token_counts = count_source_tokens(
            eligible_blocks, self._boundary_token_counters
        )
        components: list[PedagogicalComponent] = []
        cursor = 0
        while cursor < len(eligible_blocks):
            stop = min(
                cursor + self._settings.start_batch_size, len(eligible_blocks)
            )
            windows = {
                position: self._select_window(
                    eligible_blocks,
                    start_token_counts,
                    position,
                    self._settings.start_context_window,
                )
                for position in range(cursor, stop)
            }

            def project_start(
                positions: list[int],
                *,
                windows_by_position: dict[int, SourceBlockWindow] = windows,
            ) -> dict[str, object]:
                return {
                    'items': [
                        self._project_item(
                            local_position,
                            windows_by_position[candidate_position],
                        )
                        for local_position, candidate_position in enumerate(
                            positions
                        )
                    ]
                }

            batches = await pack_content_batches(
                [list(windows)],
                project=project_start,
                budget=self._start_budget,
                image_budget=self._image_budget,
            )
            batch = batches[0]
            projected = project_start(batch)
            decisions = await self._start_router.acall(**projected)
            starts = [
                batch[decision.position]
                for decision in decisions
                if decision.is_pedagogical_start
            ]
            if not starts:
                cursor = batch[-1] + 1
                continue

            start_position = min(starts)
            start_block = eligible_blocks[start_position]
            candidate = start_position + 1
            while candidate < len(eligible_blocks):
                stop = min(
                    candidate + self._settings.boundary_batch_size,
                    len(eligible_blocks),
                )
                boundary_windows = {
                    position: self._select_window(
                        eligible_blocks,
                        boundary_token_counts,
                        position,
                        self._settings.boundary_context_window,
                    )
                    for position in range(candidate, stop)
                }

                def project_boundary(
                    positions: list[int],
                    *,
                    boundary_start_block: SourceBlock = start_block,
                    windows_by_position: dict[
                        int, SourceBlockWindow
                    ] = boundary_windows,
                ) -> dict[str, object]:
                    return {
                        'start_block': project_source_block(
                            boundary_start_block
                        ),
                        'items': [
                            self._project_item(
                                local_position,
                                windows_by_position[candidate_position],
                            )
                            for local_position, candidate_position in enumerate(
                                positions
                            )
                        ],
                    }

                batches = await pack_content_batches(
                    [list(boundary_windows)],
                    project=project_boundary,
                    budget=self._boundary_budget,
                    image_budget=self._image_budget,
                )
                batch = batches[0]
                projected = project_boundary(batch)
                decisions = await self._boundary_router.acall(**projected)
                boundaries = [
                    batch[decision.position]
                    for decision in decisions
                    if decision.is_pedagogical_boundary
                ]
                if boundaries:
                    boundary = min(boundaries)
                    components.append(
                        PedagogicalComponent(
                            member_block_uuids=[
                                eligible_blocks[position].uuid
                                for position in range(start_position, boundary)
                            ]
                        )
                    )
                    cursor = boundary
                    break
                candidate = batch[-1] + 1
            else:
                components.append(
                    PedagogicalComponent(
                        member_block_uuids=[
                            eligible_blocks[position].uuid
                            for position in range(
                                start_position, len(eligible_blocks)
                            )
                        ]
                    )
                )
                cursor = len(eligible_blocks)

        return {'pedagogical_components': components}

    def _select_window(
        self,
        blocks: list[SourceBlock],
        token_counts: list[int],
        position: int,
        context_window: ContextWindowSettings,
    ) -> SourceBlockWindow:
        return select_window(
            blocks,
            [position],
            token_counts=token_counts,
            backward_budget=context_window.backward_budget,
            forward_budget=context_window.forward_budget,
            target_budget=context_window.target_budget,
        )

    @staticmethod
    def _project_item(
        position: int,
        window: SourceBlockWindow,
    ) -> PedagogicalClassificationInput:
        projected = project_source_window(window)
        return PedagogicalClassificationInput(
            position=position,
            context_before=projected.context_before,
            target_block=projected.target[0],
            context_after=projected.context_after,
        )
