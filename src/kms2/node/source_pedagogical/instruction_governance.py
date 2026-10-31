"""LangGraph node for governing exercise statements with instructions."""

import json

from kms2.config.source_pedagogical import InstructionGovernanceSettings
from kms2.core.model.source_pedagogical.instruction import (
    Instruction,
    InstructionGovernanceInput,
)
from kms2.core.windowing import (
    TextTokenCounter,
    TokenBudget,
    count_text_tokens,
    pack_items,
    project_block,
    select_window,
)
from kms2.langgraph.source_pedagogical.state import SourcePedagogicalState
from kms2.module.source_pedagogical.instruction_governance import (
    InstructionGovernanceModule,
)


class InstructionGovernanceNode:
    """Attach accepted exercise statements to their preceding instruction."""

    def __init__(
        self,
        judge: InstructionGovernanceModule,
        settings: InstructionGovernanceSettings,
        *,
        budget: TokenBudget,
        token_counters: tuple[TextTokenCounter, ...],
    ) -> None:
        self._judge = judge
        self._settings = settings
        self._budget = budget
        self._token_counters = token_counters

    async def run(
        self, state: SourcePedagogicalState
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
                window = select_window(
                    blocks,
                    member_positions,
                    token_counts=token_counts,
                    backward_budget=self._settings.context_window.backward_budget,
                    forward_budget=self._settings.context_window.forward_budget,
                    target_budget=self._settings.context_window.target_budget,
                )
                candidates.append(
                    InstructionGovernanceInput(
                        position=len(candidates),
                        context_before=window.context_before,
                        statement_blocks=window.target,
                        context_after=window.context_after,
                    )
                )
                candidate_ids.append(statement.uuid)
                local_cursor = statement_end + 1

            if candidates:
                envelope = json.dumps(
                    {
                        'instruction_blocks': [
                            block.model_dump(mode='json')
                            for block in instruction_blocks
                        ],
                        'items': [],
                    },
                    ensure_ascii=False,
                    separators=(',', ':'),
                )
                envelope_cost = self._budget.counter.count_texts([envelope])[0]
                item_payloads = [
                    json.dumps(
                        {
                            'position': self._settings.batch_size - 1,
                            'context_before': [
                                item.model_dump(mode='json')
                                for item in candidate.context_before
                            ],
                            'statement_blocks': [
                                item.model_dump(mode='json')
                                for item in candidate.statement_blocks
                            ],
                            'context_after': [
                                item.model_dump(mode='json')
                                for item in candidate.context_after
                            ],
                        },
                        ensure_ascii=False,
                        separators=(',', ':'),
                    )
                    for candidate in candidates
                ]
                item_costs = self._budget.counter.count_texts(item_payloads)
                list_overhead = self._budget.counter.count_texts(['[,]'])[0]
                batches = pack_items(
                    candidates,
                    token_counts=[cost + list_overhead for cost in item_costs],
                    token_budget=self._budget.token_limit - envelope_cost,
                    max_items=self._settings.batch_size,
                )
                governed_positions: set[int] = set()
                cursor = 0
                for batch in batches:
                    decisions = await self._judge.acall(
                        instruction_blocks=instruction_blocks,
                        items=[
                            item.model_copy(update={'position': position})
                            for position, item in enumerate(batch)
                        ],
                    )
                    governed_positions.update(
                        cursor + decision.position
                        for decision in decisions
                        if decision.governs
                    )
                    cursor += len(batch)
                governed_ids = [
                    candidate_ids[position]
                    for position in range(len(candidates))
                    if position in governed_positions
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
