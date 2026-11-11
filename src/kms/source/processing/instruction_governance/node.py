"""LangGraph node for governing exercise statements with instructions."""

from kms.core.tokenization.budget import TokenBudget
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.inference.batching import pack_content_batches
from kms.media.budget import ImagePayloadBudget
from kms.source.context import (
    SourceBlockWindow,
    count_source_tokens,
    project_source_block,
    select_window,
)
from kms.source.models.material import Instruction
from kms.source.processing.instruction_governance.inference import (
    InstructionGovernanceModule,
)
from kms.source.processing.instruction_governance.models import (
    InstructionGovernanceInput,
)
from kms.source.processing.instruction_governance.settings import (
    InstructionGovernanceSettings,
)
from kms.source.processing.state import SourceProcessingState


class InstructionGovernanceNode:
    """Attach accepted exercise statements to their preceding instruction."""

    def __init__(
        self,
        judge: InstructionGovernanceModule,
        settings: InstructionGovernanceSettings,
        *,
        budget: TokenBudget,
        token_counters: tuple[MaterialTokenEstimator, ...],
        image_budget: ImagePayloadBudget,
    ) -> None:
        self._judge = judge
        self._settings = settings
        self._budget = budget
        self._token_counters = token_counters
        self._image_budget = image_budget

    async def run(
        self, state: SourceProcessingState
    ) -> dict[str, list[Instruction]]:
        """Judge independently contextualized items within each instruction."""
        blocks = [block for page in state.split_pages for block in page.blocks]
        positions = {
            block.uuid: position for position, block in enumerate(blocks)
        }
        instruction_ranges = []
        for instruction in state.instructions:
            member_positions = [
                positions[block_uuid]
                for block_uuid in instruction.member_block_uuids
            ]
            instruction_ranges.append(
                (
                    member_positions[0],
                    member_positions[-1],
                    instruction,
                    member_positions,
                )
            )
        instruction_ranges.sort(key=lambda item: item[0])
        statement_ranges = []
        for statement in state.statements:
            if not statement.is_exercise:
                continue
            member_positions = [
                positions[block_uuid]
                for block_uuid in statement.member_block_uuids
            ]
            statement_ranges.append(
                (member_positions[0], member_positions[-1], statement)
            )
        statement_ranges.sort(key=lambda item: item[0])

        eligible = [
            (
                statement_start,
                statement_end,
                statement,
            )
            for instruction_index, (
                _instruction_start,
                instruction_end,
                _instruction,
                _instruction_positions,
            ) in enumerate(instruction_ranges)
            for next_instruction_start in [
                instruction_ranges[instruction_index + 1][0]
                if instruction_index + 1 < len(instruction_ranges)
                else len(blocks)
            ]
            for statement_start, statement_end, statement in statement_ranges
            if instruction_end < statement_start < next_instruction_start
        ]
        token_counts = []
        if eligible:
            token_counts = count_source_tokens(blocks, self._token_counters)
        governed_instructions: list[Instruction] = []
        statement_cursor = 0
        for instruction_index, (
            _instruction_start,
            instruction_end,
            instruction,
            instruction_positions,
        ) in enumerate(instruction_ranges):
            next_instruction_start = len(blocks)
            if instruction_index + 1 < len(instruction_ranges):
                next_instruction_start = instruction_ranges[
                    instruction_index + 1
                ][0]
            local_cursor = max(statement_cursor, instruction_end + 1)
            candidates = []
            candidate_ids = []
            for statement_start, statement_end, statement in statement_ranges:
                if statement_start < local_cursor:
                    continue
                if statement_start >= next_instruction_start:
                    break
                member_positions = [
                    positions[block_uuid]
                    for block_uuid in statement.member_block_uuids
                ]
                candidates.append(
                    select_window(
                        blocks,
                        member_positions,
                        token_counts=token_counts,
                        backward_budget=self._settings.context_window.backward_budget,
                        forward_budget=self._settings.context_window.forward_budget,
                        target_budget=self._settings.context_window.target_budget,
                    )
                )
                candidate_ids.append(statement.uuid)
                local_cursor = statement_end + 1

            if candidates:

                def project(
                    indexes: list[int],
                    *,
                    instruction_positions_by_window: list[
                        int
                    ] = instruction_positions,
                    candidate_windows: list[SourceBlockWindow] = candidates,
                ) -> dict[str, object]:
                    return {
                        'instruction_blocks': [
                            project_source_block(blocks[position])
                            for position in instruction_positions_by_window
                        ],
                        'items': [
                            InstructionGovernanceInput(
                                position=position,
                                context_before=[
                                    project_source_block(block)
                                    for block in candidate_windows[
                                        index
                                    ].context_before
                                ],
                                statement_blocks=[
                                    project_source_block(block)
                                    for block in candidate_windows[index].target
                                ],
                                context_after=[
                                    project_source_block(block)
                                    for block in candidate_windows[
                                        index
                                    ].context_after
                                ],
                            )
                            for position, index in enumerate(indexes)
                        ],
                    }

                incoming_batches = [
                    list(
                        range(
                            start,
                            min(
                                start + self._settings.batch_size,
                                len(candidates),
                            ),
                        )
                    )
                    for start in range(
                        0, len(candidates), self._settings.batch_size
                    )
                ]
                batches = await pack_content_batches(
                    incoming_batches,
                    project=project,
                    budget=self._budget,
                    image_budget=self._image_budget,
                )
                governed_positions: set[int] = set()
                for batch in batches:
                    projected = project(batch)
                    decisions = await self._judge.acall(**projected)
                    governed_positions.update(
                        batch[decision.position]
                        for decision in decisions
                        if decision.governs
                    )
                governed_ids = [
                    candidate_ids[position]
                    for position in sorted(governed_positions)
                ]
            else:
                governed_ids = []
            statement_cursor = local_cursor
            governed_instructions.append(
                instruction.model_copy(
                    update={'governed_statement_uuids': governed_ids}
                )
            )

        return {'instructions': governed_instructions}
