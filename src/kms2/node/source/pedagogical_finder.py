"""LangGraph node for discovering pedagogical source components."""

import logging

from kms2.config.source import PedagogicalFinderSettings
from kms2.core.model import PedagogicalComponent
from kms2.core.windowing import select_window
from kms2.langgraph.source.state import SourceState
from kms2.module.source.pedagogical_finder import (
    PedagogicalBoundaryRouterModule,
    PedagogicalStartRouterModule,
)

logger = logging.getLogger(__name__)


def _excerpt(content: str | None) -> str:
    """Return a compact source excerpt for boundary diagnostics."""
    return ' '.join((content or '').split())[:160]


class PedagogicalFinderNode:
    """Find ordered pedagogical components outside instruction members."""

    def __init__(
        self,
        start_router: PedagogicalStartRouterModule,
        boundary_router: PedagogicalBoundaryRouterModule,
        settings: PedagogicalFinderSettings,
    ) -> None:
        self._start_router = start_router
        self._boundary_router = boundary_router
        self._settings = settings

    async def run(
        self, state: SourceState
    ) -> dict[str, list[PedagogicalComponent]]:
        """Discover pedagogical components without modifying source pages."""
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
        eligible_positions = [
            position
            for position, block in enumerate(blocks)
            if block.uuid not in claimed_members
        ]
        eligible_blocks = [blocks[position] for position in eligible_positions]
        components: list[PedagogicalComponent] = []
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
            logger.info(
                'pedagogical start cursor=%d eligible_position=%d decision=%s content=%r',
                cursor,
                eligible_positions[cursor],
                is_start,
                _excerpt(start_window.target[0].content),
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
                logger.info(
                    'pedagogical boundary start_cursor=%d candidate=%d eligible_position=%d decision=%s content=%r',
                    cursor,
                    candidate,
                    eligible_positions[candidate],
                    is_boundary,
                    _excerpt(boundary_window.target[0].content),
                )
                if is_boundary:
                    components.append(
                        PedagogicalComponent(
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
                    PedagogicalComponent(
                        member_block_uuids=[
                            eligible_blocks[position].uuid
                            for position in range(cursor, len(eligible_blocks))
                        ]
                    )
                )
                cursor = len(eligible_blocks)

        return {'pedagogical_components': components}
