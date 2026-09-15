"""LangGraph node for discovering shared source instructions."""

from kms2.config import InstructionFinderSettings
from kms2.core.model import Instruction
from kms2.core.windowing import select_window
from kms2.langgraph.source.state import SourceState
from kms2.module.source.instruction_finder import (
    InstructionBoundaryRouterModule,
    InstructionStartRouterModule,
)


class InstructionFinderNode:
    """Find ordered instruction pointers without mutating source blocks."""

    def __init__(
        self,
        start_router: InstructionStartRouterModule,
        boundary_router: InstructionBoundaryRouterModule,
        settings: InstructionFinderSettings,
    ) -> None:
        self._start_router = start_router
        self._boundary_router = boundary_router
        self._settings = settings

    async def run(self, state: SourceState) -> dict[str, list[Instruction]]:
        """Discover shared instructions across the final split source stream."""
        blocks = [block for page in state.split_pages for block in page.blocks]
        instructions: list[Instruction] = []
        cursor = 0
        while cursor < len(blocks):
            start_window = select_window(
                blocks,
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
            while candidate < len(blocks):
                boundary_window = select_window(
                    blocks,
                    [candidate],
                    backward_budget=(
                        self._settings.boundary_context_window.backward_budget
                    ),
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
                    instructions.append(
                        Instruction(
                            member_block_uuids=[
                                blocks[position].uuid
                                for position in range(cursor, candidate)
                            ]
                        )
                    )
                    cursor = candidate
                    break
                candidate += 1
            else:
                instructions.append(
                    Instruction(
                        member_block_uuids=[
                            blocks[position].uuid
                            for position in range(cursor, len(blocks))
                        ]
                    )
                )
                cursor = len(blocks)

        return {'instructions': instructions}
