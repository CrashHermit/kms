"""LangGraph node for constructing statement and procedure pointers."""

from kms2.core.model import Procedure, Statement
from kms2.core.model.source_stage.pedagogical import PedagogicalMember
from kms2.core.windowing import project_block
from kms2.langgraph.source.state import SourceState
from kms2.module.source.statement_procedure import (
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
    ) -> None:
        self._role_typer = role_typer
        self._statement_partitioner = statement_partitioner
        self._procedure_partitioner = procedure_partitioner

    async def run(
        self, state: SourceState
    ) -> dict[str, list[Statement] | list[Procedure]]:
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
            Statement(
                member_block_uuids=component.member_block_uuids,
                is_exercise=True,
            )
            for component in state.exercise_components
        ]
        procedures: list[Procedure] = []

        for component in state.pedagogical_components:
            members = [
                PedagogicalMember(
                    position=position,
                    source_block=project_block(blocks_by_uuid[block_uuid]),
                )
                for position, block_uuid in enumerate(
                    component.member_block_uuids
                )
            ]
            has_statement, has_procedure = await self._role_typer.aforward(
                members=members
            )
            if not has_statement and not has_procedure:
                continue
            if has_statement and not has_procedure:
                statements.append(
                    Statement(member_block_uuids=component.member_block_uuids)
                )
                continue
            if has_procedure and not has_statement:
                procedures.append(
                    Procedure(member_block_uuids=component.member_block_uuids)
                )
                continue

            statement_positions = await self._statement_partitioner.aforward(
                members=members
            )
            procedure_positions = await self._procedure_partitioner.aforward(
                members=members
            )
            statements.append(
                Statement(
                    member_block_uuids=[
                        component.member_block_uuids[position]
                        for position in statement_positions
                    ]
                )
            )
            procedures.append(
                Procedure(
                    member_block_uuids=[
                        component.member_block_uuids[position]
                        for position in procedure_positions
                    ]
                )
            )

        statements.sort(
            key=lambda statement: block_positions[
                statement.member_block_uuids[0]
            ]
        )
        return {'statements': statements, 'procedures': procedures}
