"""LangGraph node for constructing statement and procedure pointers."""

import json

from kms2.config.source_pedagogical import StatementProcedureSettings
from kms2.core.model.source_pedagogical.pedagogical import (
    PedagogicalComponentInput,
    PedagogicalMember,
    PedagogicalRoleDecision,
    ProcedureDraft,
    ProcedurePartitionDecision,
    StatementDraft,
    StatementPartitionDecision,
)
from kms2.core.windowing import TokenBudget, pack_items, project_block
from kms2.langgraph.source_pedagogical.state import SourcePedagogicalState
from kms2.module.source_pedagogical.statement_procedure import (
    PedagogicalRoleModule,
    ProcedurePartitionModule,
    StatementPartitionModule,
)


class StatementProcedureNode:
    """Build ordered statement and procedure pointers per component."""

    def __init__(
        self,
        role_typer: PedagogicalRoleModule,
        statement_partitioner: StatementPartitionModule,
        procedure_partitioner: ProcedurePartitionModule,
        settings: StatementProcedureSettings,
        *,
        role_budget: TokenBudget,
        statement_budget: TokenBudget,
        procedure_budget: TokenBudget,
    ) -> None:
        self._role_typer = role_typer
        self._statement_partitioner = statement_partitioner
        self._procedure_partitioner = procedure_partitioner
        self._settings = settings
        self._role_budget = role_budget
        self._statement_budget = statement_budget
        self._procedure_budget = procedure_budget

    @staticmethod
    def _pack(
        items: list[PedagogicalComponentInput],
        *,
        budget: TokenBudget,
        batch_size: int,
    ) -> list[list[PedagogicalComponentInput]]:
        item_costs = budget.counter.count_texts(
            [
                json.dumps(
                    {
                        'position': batch_size - 1,
                        'members': [
                            {
                                'position': member.position,
                                'source_block': member.source_block.model_dump(
                                    mode='json'
                                ),
                            }
                            for member in item.members
                        ],
                    },
                    ensure_ascii=False,
                    separators=(',', ':'),
                )
                for item in items
            ]
        )
        list_overhead = budget.counter.count_texts(['[,]'])[0]
        envelope_cost = budget.counter.count_texts(['{"items":[]}'])[0]
        return pack_items(
            items,
            token_counts=[cost + list_overhead for cost in item_costs],
            token_budget=budget.token_limit - envelope_cost,
            max_items=batch_size,
        )

    async def _classify(
        self,
        items: list[PedagogicalComponentInput],
    ) -> list[PedagogicalRoleDecision]:
        decisions: list[PedagogicalRoleDecision] = []
        cursor = 0
        for batch in self._pack(
            items,
            budget=self._role_budget,
            batch_size=self._settings.role_batch_size,
        ):
            batch_decisions = await self._role_typer.acall(
                items=[
                    item.model_copy(update={'position': position})
                    for position, item in enumerate(batch)
                ]
            )
            decisions.extend(
                decision.model_copy(
                    update={'position': cursor + decision.position}
                )
                for decision in batch_decisions
            )
            cursor += len(batch)
        return decisions

    async def _partition(
        self,
        items: list[PedagogicalComponentInput],
        *,
        budget: TokenBudget,
        batch_size: int,
        module: StatementPartitionModule | ProcedurePartitionModule,
    ) -> list[StatementPartitionDecision | ProcedurePartitionDecision]:
        decisions: list[
            StatementPartitionDecision | ProcedurePartitionDecision
        ] = []
        cursor = 0
        for batch in self._pack(items, budget=budget, batch_size=batch_size):
            batch_decisions = await module.acall(
                items=[
                    item.model_copy(update={'position': position})
                    for position, item in enumerate(batch)
                ]
            )
            decisions.extend(
                decision.model_copy(
                    update={'position': cursor + decision.position}
                )
                for decision in batch_decisions
            )
            cursor += len(batch)
        return decisions

    async def run(
        self, state: SourcePedagogicalState
    ) -> dict[str, list[StatementDraft] | list[ProcedureDraft]]:
        """Construct pointers without mutating canonical source blocks."""
        blocks_by_uuid = {
            block.uuid: block
            for page in state.split_pages
            for block in page.blocks
        }
        block_positions = {
            block.uuid: position
            for position, block in enumerate(
                block for page in state.split_pages for block in page.blocks
            )
        }
        statements = [
            StatementDraft(
                member_block_uuids=component.member_block_uuids,
                is_exercise=True,
            )
            for component in state.exercise_components
        ]
        procedures: list[ProcedureDraft] = []
        components = [
            PedagogicalComponentInput(
                position=position,
                members=[
                    PedagogicalMember(
                        position=member_position,
                        source_block=project_block(blocks_by_uuid[block_uuid]),
                    )
                    for member_position, block_uuid in enumerate(
                        component.member_block_uuids
                    )
                ],
            )
            for position, component in enumerate(state.pedagogical_components)
        ]
        if components:
            roles = self._classify(components)
            role_decisions = await roles
        else:
            role_decisions = []
        role_by_position = {
            decision.position: decision for decision in role_decisions
        }
        mixed: list[PedagogicalComponentInput] = [
            components[position]
            for position in range(len(components))
            if role_by_position[position].has_statement
            and role_by_position[position].has_procedure
        ]

        if mixed:
            statement_decisions = await self._partition(
                mixed,
                budget=self._statement_budget,
                batch_size=self._settings.statement_batch_size,
                module=self._statement_partitioner,
            )
            procedure_decisions = await self._partition(
                mixed,
                budget=self._procedure_budget,
                batch_size=self._settings.procedure_batch_size,
                module=self._procedure_partitioner,
            )
        else:
            statement_decisions = []
            procedure_decisions = []
        statement_by_position = {
            decision.position: decision for decision in statement_decisions
        }
        procedure_by_position = {
            decision.position: decision for decision in procedure_decisions
        }
        mixed_position_by_component = {
            item.position: position for position, item in enumerate(mixed)
        }
        for position, component in enumerate(state.pedagogical_components):
            decision = role_by_position[position]
            if decision.has_statement and decision.has_procedure:
                mixed_position = mixed_position_by_component[position]
                statement_positions = statement_by_position[
                    mixed_position
                ].statement_positions
                procedure_positions = procedure_by_position[
                    mixed_position
                ].procedure_positions
                statements.append(
                    StatementDraft(
                        member_block_uuids=[
                            component.member_block_uuids[member_position]
                            for member_position in statement_positions
                        ]
                    )
                )
                procedures.append(
                    ProcedureDraft(
                        member_block_uuids=[
                            component.member_block_uuids[member_position]
                            for member_position in procedure_positions
                        ]
                    )
                )
            elif decision.has_statement:
                statements.append(
                    StatementDraft(
                        member_block_uuids=component.member_block_uuids
                    )
                )
            elif decision.has_procedure:
                procedures.append(
                    ProcedureDraft(
                        member_block_uuids=component.member_block_uuids
                    )
                )

        statements.sort(
            key=lambda statement: block_positions[
                statement.member_block_uuids[0]
            ]
        )
        return {'statements': statements, 'procedures': procedures}
