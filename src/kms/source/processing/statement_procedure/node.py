"""LangGraph node for constructing statement and procedure pointers."""

from kms.core.tokenization.budget import TokenBudget
from kms.inference.batching import pack_content_batches
from kms.media.budget import ImagePayloadBudget
from kms.source.context import project_source_block
from kms.source.models.material import (
    PedagogicalComponent,
    ProcedureDraft,
    StatementDraft,
)
from kms.source.models.source import SourceBlock
from kms.source.processing.state import SourceProcessingState
from kms.source.processing.statement_procedure.inference import (
    PedagogicalRoleModule,
    ProcedurePartitionModule,
    StatementPartitionModule,
)
from kms.source.processing.statement_procedure.models import (
    PedagogicalComponentInput,
    PedagogicalMember,
    PedagogicalRoleDecision,
    ProcedurePartitionDecision,
    StatementPartitionDecision,
)
from kms.source.processing.statement_procedure.settings import (
    StatementProcedureSettings,
)


class StatementProcedurePartitionNode:
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
    def _project(
        positions: list[int],
        components: list[PedagogicalComponent],
        blocks_by_uuid: dict[str, SourceBlock],
    ) -> list[PedagogicalComponentInput]:
        """Project canonical member blocks for one model call."""
        return [
            PedagogicalComponentInput(
                position=position,
                members=[
                    PedagogicalMember(
                        position=member_position,
                        source_block=project_source_block(
                            blocks_by_uuid[block_uuid]
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
        positions = list(range(len(components)))
        incoming = [
            positions[start : start + self._settings.role_batch_size]
            for start in range(
                0, len(positions), self._settings.role_batch_size
            )
        ]

        def project(indexes: list[int]) -> dict[str, object]:
            return {'items': self._project(indexes, components, blocks_by_uuid)}

        batches = await pack_content_batches(
            incoming,
            project=project,
            budget=self._role_budget,
            image_budget=self._image_budget,
        )
        decisions: list[PedagogicalRoleDecision] = []
        for actual_positions in batches:
            projected = project(actual_positions)
            batch_decisions = await self._role_typer.acall(**projected)
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
        local_positions = list(range(len(component_positions)))
        incoming = [
            local_positions[start : start + batch_size]
            for start in range(0, len(local_positions), batch_size)
        ]

        def project(indexes: list[int]) -> dict[str, object]:
            return {
                'items': self._project(
                    [component_positions[position] for position in indexes],
                    components,
                    blocks_by_uuid,
                )
            }

        batches = await pack_content_batches(
            incoming,
            project=project,
            budget=budget,
            image_budget=self._image_budget,
        )
        decisions: list[
            StatementPartitionDecision | ProcedurePartitionDecision
        ] = []
        for actual_positions in batches:
            batch_decisions = await module.acall(**project(actual_positions))
            decisions.extend(
                decision.model_copy(
                    update={'position': actual_positions[decision.position]}
                )
                for decision in batch_decisions
            )
        return decisions

    async def run(
        self, state: SourceProcessingState
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
