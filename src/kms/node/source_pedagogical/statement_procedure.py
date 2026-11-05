"""LangGraph node for constructing statement and procedure pointers."""

import json

import dspy

from kms.config.source_pedagogical import StatementProcedureSettings
from kms.core.content import MultimodalContentBuilder
from kms.core.media_budget import ImagePayloadBudget
from kms.core.model.block import SourceBlock
from kms.core.model.source_pedagogical.pedagogical import (
    PedagogicalComponent,
    PedagogicalComponentInput,
    PedagogicalMember,
    PedagogicalRoleDecision,
    ProcedureDraft,
    ProcedurePartitionDecision,
    StatementDraft,
    StatementPartitionDecision,
)
from kms.core.source_context import project_source_block
from kms.core.tokenization.budget import TokenBudget
from kms.core.windowing import pack_items
from kms.langgraph.source_pedagogical.state import SourcePedagogicalState
from kms.module.source_pedagogical.statement_procedure import (
    PedagogicalRoleModule,
    ProcedurePartitionModule,
    StatementPartitionModule,
)
from kms.service.content import (
    load_content_images,
    pack_content_batches,
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
        image_budget: ImagePayloadBudget,
    ) -> None:
        self._role_typer = role_typer
        self._statement_partitioner = statement_partitioner
        self._procedure_partitioner = procedure_partitioner
        self._settings = settings
        self._role_budget = role_budget
        self._statement_budget = statement_budget
        self._procedure_budget = procedure_budget
        self._image_budget = image_budget

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

    @staticmethod
    def _project(
        positions: list[int],
        components: list[PedagogicalComponent],
        blocks_by_uuid: dict[str, SourceBlock],
        builder: MultimodalContentBuilder,
    ) -> list[PedagogicalComponentInput]:
        """Project canonical member blocks for one model call."""
        return [
            PedagogicalComponentInput(
                position=position,
                members=[
                    PedagogicalMember(
                        position=member_position,
                        source_block=project_source_block(
                            blocks_by_uuid[block_uuid], builder
                        ),
                    )
                    for member_position, block_uuid in enumerate(
                        components[component_position].member_block_uuids
                    )
                ],
            )
            for position, component_position in enumerate(positions)
        ]

    async def _classify(
        self,
        components: list[PedagogicalComponent],
        blocks_by_uuid: dict[str, SourceBlock],
    ) -> list[PedagogicalRoleDecision]:
        estimates = self._project(
            list(range(len(components))),
            components,
            blocks_by_uuid,
            MultimodalContentBuilder(),
        )
        decisions: list[PedagogicalRoleDecision] = []
        for batch in self._pack(
            estimates,
            budget=self._role_budget,
            batch_size=self._settings.role_batch_size,
        ):
            positions = [item.position for item in batch]

            def project(
                indexes: list[int], builder: MultimodalContentBuilder
            ) -> dict[str, object]:
                return {
                    'items': self._project(
                        indexes, components, blocks_by_uuid, builder
                    )
                }

            actual_batches = await pack_content_batches(
                [positions],
                project=project,
                budget=self._role_budget,
                image_budget=self._image_budget,
            )
            for actual_positions in actual_batches:
                builder = MultimodalContentBuilder()
                projected = project(actual_positions, builder)
                payloads = await load_content_images(
                    builder.images, budget=self._image_budget
                )
                batch_decisions = await self._role_typer.acall(
                    **projected,
                    content_images=[
                        dspy.Image(payload) for payload in payloads
                    ],
                )
                decisions.extend(
                    decision.model_copy(
                        update={'position': actual_positions[decision.position]}
                    )
                    for decision in batch_decisions
                )
        return decisions

    async def _partition(
        self,
        component_positions: list[int],
        components: list[PedagogicalComponent],
        blocks_by_uuid: dict[str, SourceBlock],
        *,
        budget: TokenBudget,
        batch_size: int,
        module: StatementPartitionModule | ProcedurePartitionModule,
    ) -> list[StatementPartitionDecision | ProcedurePartitionDecision]:
        estimates = self._project(
            component_positions,
            components,
            blocks_by_uuid,
            MultimodalContentBuilder(),
        )
        decisions: list[
            StatementPartitionDecision | ProcedurePartitionDecision
        ] = []
        for batch in self._pack(
            estimates, budget=budget, batch_size=batch_size
        ):
            local_positions = [item.position for item in batch]

            def project(
                indexes: list[int], builder: MultimodalContentBuilder
            ) -> dict[str, object]:
                return {
                    'items': self._project(
                        [component_positions[position] for position in indexes],
                        components,
                        blocks_by_uuid,
                        builder,
                    )
                }

            actual_batches = await pack_content_batches(
                [local_positions],
                project=project,
                budget=budget,
                image_budget=self._image_budget,
            )
            for actual_positions in actual_batches:
                canonical_positions = [
                    component_positions[position]
                    for position in actual_positions
                ]
                builder = MultimodalContentBuilder()
                items = self._project(
                    canonical_positions, components, blocks_by_uuid, builder
                )
                payloads = await load_content_images(
                    builder.images, budget=self._image_budget
                )
                batch_decisions = await module.acall(
                    items=items,
                    content_images=[
                        dspy.Image(payload) for payload in payloads
                    ],
                )
                decisions.extend(
                    decision.model_copy(
                        update={'position': actual_positions[decision.position]}
                    )
                    for decision in batch_decisions
                )
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
        components = state.pedagogical_components
        role_decisions = (
            await self._classify(components, blocks_by_uuid)
            if components
            else []
        )
        role_by_position = {
            decision.position: decision for decision in role_decisions
        }
        mixed_source_positions = [
            position
            for position in range(len(components))
            if role_by_position[position].has_statement
            and role_by_position[position].has_procedure
        ]

        if mixed_source_positions:
            statement_decisions = await self._partition(
                mixed_source_positions,
                components,
                blocks_by_uuid,
                budget=self._statement_budget,
                batch_size=self._settings.statement_batch_size,
                module=self._statement_partitioner,
            )
            procedure_decisions = await self._partition(
                mixed_source_positions,
                components,
                blocks_by_uuid,
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
            component_position: mixed_position
            for mixed_position, component_position in enumerate(
                mixed_source_positions
            )
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
