import asyncio

from kms2.core.model import (
    BlockType,
    ExerciseComponent,
    PedagogicalComponent,
    Source,
    SourceBlock,
    SourcePage,
)
from kms2.langgraph.source.state import SourceState
from kms2.node.source.statement_procedure import StatementProcedureNode


class RoleTyper:
    def __init__(self) -> None:
        self.calls = []

    async def aforward(self, *, members):
        self.calls.append(members)
        return True, True


class StatementPartitioner:
    async def aforward(self, *, members):
        assert [member.position for member in members] == [0, 1, 2]
        return [0, 2]


class ProcedurePartitioner:
    async def aforward(self, *, members):
        assert [member.position for member in members] == [0, 1, 2]
        return [1]


class FixedRoleTyper:
    def __init__(self, result):
        self.result = result

    async def aforward(self, *, members):
        return self.result


def _state() -> SourceState:
    blocks = [
        SourceBlock(
            uuid=f'block-{index}',
            block_type=BlockType.PARAGRAPH,
            content=str(index),
        )
        for index in range(3)
    ]
    return SourceState(
        pdf_path='source.pdf',
        source=Source(key='source-1'),
        split_pages=[SourcePage(index=0, blocks=blocks)],
        pedagogical_components=[
            PedagogicalComponent(
                member_block_uuids=['block-0', 'block-1', 'block-2']
            )
        ],
    )


def _interleaved_state() -> SourceState:
    blocks = [
        SourceBlock(
            uuid=f'block-{index}',
            block_type=BlockType.PARAGRAPH,
            content=str(index),
        )
        for index in range(4)
    ]
    return SourceState(
        pdf_path='source.pdf',
        source=Source(key='source-1'),
        split_pages=[SourcePage(index=0, blocks=blocks)],
        exercise_components=[ExerciseComponent(member_block_uuids=['block-0'])],
        pedagogical_components=[
            PedagogicalComponent(
                member_block_uuids=['block-1', 'block-2', 'block-3']
            )
        ],
    )


def test_statement_procedure_materializes_exercises_before_role_partitioning():
    role_typer = RoleTyper()

    result = asyncio.run(
        StatementProcedureNode(
            role_typer, StatementPartitioner(), ProcedurePartitioner()
        ).run(_interleaved_state())
    )

    assert [
        (statement.member_block_uuids, statement.is_exercise)
        for statement in result['statements']
    ] == [
        (['block-0'], True),
        (['block-1', 'block-3'], False),
    ]
    assert [
        procedure.member_block_uuids for procedure in result['procedures']
    ] == [['block-2']]
    assert len(role_typer.calls) == 1


def test_statement_procedure_maps_independent_zero_based_partitions():
    state = _state()
    original_pages = state.split_pages

    result = asyncio.run(
        StatementProcedureNode(
            RoleTyper(), StatementPartitioner(), ProcedurePartitioner()
        ).run(state)
    )

    assert result['statements'][0].member_block_uuids == ['block-0', 'block-2']
    assert result['procedures'][0].member_block_uuids == ['block-1']
    assert state.split_pages is original_pages


def test_statement_procedure_role_branches_create_only_selected_pointer():
    for role, expected in [
        ((True, False), ('statements', 1)),
        ((False, True), ('procedures', 1)),
        ((False, False), ('statements', 0)),
    ]:
        result = asyncio.run(
            StatementProcedureNode(
                FixedRoleTyper(role),
                StatementPartitioner(),
                ProcedurePartitioner(),
            ).run(_state())
        )
        assert len(result[expected[0]]) == expected[1]
        other = 'procedures' if expected[0] == 'statements' else 'statements'
        assert result[other] == []
