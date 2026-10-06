"""LangGraph node for governing exercise statements with instructions."""

from kms2.config.source_processing import InstructionGovernanceSettings
from kms2.core.model.source_processing.instruction import Instruction
from kms2.core.windowing import (
    TextTokenCounter,
    count_text_tokens,
    project_block,
    select_window,
)
from kms2.langgraph.source_processing.state import SourceProcessingState
from kms2.module.source_processing.instruction_governance import (
    InstructionGovernanceModule,
)


class InstructionGovernanceNode:
    """Attach accepted exercise statements to their preceding instruction."""

    def __init__(
        self,
        judge: InstructionGovernanceModule,
        settings: InstructionGovernanceSettings,
        *,
        token_counters: tuple[TextTokenCounter, ...],
    ) -> None:
        self._judge = judge
        self._settings = settings
        self._token_counters = token_counters

    async def run(
        self, state: SourceProcessingState
    ) -> dict[str, list[Instruction]]:
        """Judge exercise statements within each instruction domain."""
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
        has_eligible_statement = any(
            instruction_end < statement_start < next_instruction_start
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
            for statement_start, _statement_end, _statement in statement_ranges
        )
        token_counts = []
        if instruction_ranges and has_eligible_statement:
            token_counts = count_text_tokens(
                [block.content for block in blocks], self._token_counters
            )
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
            instruction_blocks = [
                project_block(blocks[position])
                for position in instruction_positions
            ]
            governed_statement_uuids: list[str] = []
            for statement_start, statement_end, statement in statement_ranges:
                if statement_start < local_cursor:
                    continue
                if statement_start >= next_instruction_start:
                    break

                statement_positions = [
                    positions[block_uuid]
                    for block_uuid in statement.member_block_uuids
                ]
                statement_window = select_window(
                    blocks,
                    statement_positions,
                    token_counts=token_counts,
                    backward_budget=self._settings.context_window.backward_budget,
                    forward_budget=self._settings.context_window.forward_budget,
                    target_budget=self._settings.context_window.target_budget,
                )
                governs = await self._judge.aforward(
                    instruction_blocks=instruction_blocks,
                    context_before=statement_window.context_before,
                    statement_blocks=statement_window.target,
                    context_after=statement_window.context_after,
                )
                if governs:
                    governed_statement_uuids.append(statement.uuid)
                local_cursor = statement_end + 1

            statement_cursor = local_cursor
            governed_instructions.append(
                instruction.model_copy(
                    update={
                        'governed_statement_uuids': governed_statement_uuids
                    }
                )
            )

        return {'instructions': governed_instructions}
