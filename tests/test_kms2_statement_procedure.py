import asyncio

from kms2.config.source_pedagogical import StatementProcedureSettings
from kms2.core.model.block import SourceBlock
from kms2.core.model.block_types import BlockType
from kms2.core.model.page import SourcePage
from kms2.core.model.source_pedagogical.pedagogical import (
    ExerciseComponent,
    PedagogicalComponent,
    PedagogicalComponentInput,
    PedagogicalRoleDecision,
    ProcedurePartitionDecision,
    StatementPartitionDecision,
)
from kms2.core.windowing import TokenBudget
from kms2.langgraph.source_pedagogical.state import SourcePedagogicalState
from kms2.node.source_pedagogical.statement_procedure import (
    StatementProcedureNode,
)


class _Counter:
    def count_texts(self, texts: list[str]) -> list[int]:
        return [1 for _ in texts]


class _RoleTyper:
    def __init__(self) -> None:
        self.calls: list[list[PedagogicalComponentInput]] = []

    async def acall(
        self, *, items: list[PedagogicalComponentInput]
    ) -> list[PedagogicalRoleDecision]:
        self.calls.append(items)
        decisions = []
        for item in items:
            label = (
                item.members[0].source_block.content.split()[0].split('-')[0]
            )
            roles = {
                'statement': (True, False),
                'procedure': (False, True),
                'neither': (False, False),
                'mixed': (True, True),
            }[label]
            decisions.append(
                PedagogicalRoleDecision(
                    position=item.position,
                    has_statement=roles[0],
                    has_procedure=roles[1],
                )
            )
        return list(reversed(decisions))


class _StatementPartitioner:
    def __init__(self) -> None:
        self.calls: list[list[PedagogicalComponentInput]] = []

    async def acall(
        self, *, items: list[PedagogicalComponentInput]
    ) -> list[StatementPartitionDecision]:
        self.calls.append(items)
        return [
            StatementPartitionDecision(
                position=item.position,
                statement_positions=list(range(len(item.members))),
            )
            for item in reversed(items)
        ]


class _ProcedurePartitioner:
    def __init__(self) -> None:
        self.calls: list[list[PedagogicalComponentInput]] = []

    async def acall(
        self, *, items: list[PedagogicalComponentInput]
    ) -> list[ProcedurePartitionDecision]:
        self.calls.append(items)
        return [
            ProcedurePartitionDecision(
                position=item.position,
                procedure_positions=list(range(len(item.members))),
            )
            for item in reversed(items)
        ]


def _state() -> SourcePedagogicalState:
    labels = [
        'exercise',
        'statement-a first',
        'statement-a second',
        'procedure-a',
        'neither-a',
        'mixed-a first',
        'mixed-a second',
        'mixed-a third',
        'mixed-b first',
        'mixed-b second',
    ]
    blocks = [
        SourceBlock(
            uuid=f'block-{index}',
            block_type=BlockType.PARAGRAPH,
            content=content,
        )
        for index, content in enumerate(labels)
    ]
    return SourcePedagogicalState(
        source_uuid='source-1',
        split_pages=[SourcePage(index=0, blocks=blocks)],
        exercise_components=[ExerciseComponent(member_block_uuids=['block-0'])],
        pedagogical_components=[
            PedagogicalComponent(member_block_uuids=['block-1', 'block-2']),
            PedagogicalComponent(member_block_uuids=['block-3']),
            PedagogicalComponent(member_block_uuids=['block-4']),
            PedagogicalComponent(
                member_block_uuids=['block-5', 'block-6', 'block-7']
            ),
            PedagogicalComponent(member_block_uuids=['block-8', 'block-9']),
        ],
    )


def _run(
    batch_size: int,
    state: SourcePedagogicalState | None = None,
) -> tuple[dict, _RoleTyper, _StatementPartitioner, _ProcedurePartitioner]:
    state = state or _state()
    role_typer = _RoleTyper()
    statement_partitioner = _StatementPartitioner()
    procedure_partitioner = _ProcedurePartitioner()
    settings = StatementProcedureSettings(
        role_batch_size=batch_size,
        statement_batch_size=batch_size,
        procedure_batch_size=batch_size,
    )
    node = StatementProcedureNode(
        role_typer,
        statement_partitioner,
        procedure_partitioner,
        settings,
        role_budget=TokenBudget(_Counter(), 10_000),
        statement_budget=TokenBudget(_Counter(), 10_000),
        procedure_budget=TokenBudget(_Counter(), 10_000),
    )
    return (
        asyncio.run(node.run(state)),
        role_typer,
        statement_partitioner,
        procedure_partitioner,
    )


def test_component_batches_preserve_roles_partition_scopes_and_order():
    state = _state()
    original_pages = [page.model_copy(deep=True) for page in state.split_pages]
    result, roles, statements, procedures = _run(16, state)

    assert len(roles.calls) == 1
    assert len(statements.calls) == 1
    assert len(procedures.calls) == 1
    assert len(roles.calls[0]) == 5
    assert all(
        [member.position for member in item.members]
        == list(range(len(item.members)))
        for item in roles.calls[0]
    )
    assert [
        (item.member_block_uuids, item.is_exercise)
        for item in result['statements']
    ] == [
        (['block-0'], True),
        (['block-1', 'block-2'], False),
        (['block-5', 'block-6', 'block-7'], False),
        (['block-8', 'block-9'], False),
    ]
    assert [item.member_block_uuids for item in result['procedures']] == [
        ['block-3'],
        ['block-5', 'block-6', 'block-7'],
        ['block-8', 'block-9'],
    ]
    assert set(result['statements'][2].member_block_uuids) == set(
        result['procedures'][1].member_block_uuids
    )
    assert state.split_pages == original_pages

    singleton, single_roles, single_statements, single_procedures = _run(1)
    assert len(single_roles.calls) == 5
    assert len(single_statements.calls) == 2
    assert len(single_procedures.calls) == 2
    assert [item.member_block_uuids for item in singleton['statements']] == [
        item.member_block_uuids for item in result['statements']
    ]
    assert [item.member_block_uuids for item in singleton['procedures']] == [
        item.member_block_uuids for item in result['procedures']
    ]


def test_no_pedagogical_components_skips_role_and_partition_calls():
    state = _state()
    state.pedagogical_components = []
    role_typer = _RoleTyper()
    statement_partitioner = _StatementPartitioner()
    procedure_partitioner = _ProcedurePartitioner()
    settings = StatementProcedureSettings()
    result = asyncio.run(
        StatementProcedureNode(
            role_typer,
            statement_partitioner,
            procedure_partitioner,
            settings,
            role_budget=TokenBudget(_Counter(), 10_000),
            statement_budget=TokenBudget(_Counter(), 10_000),
            procedure_budget=TokenBudget(_Counter(), 10_000),
        ).run(state)
    )
    assert [
        (item.member_block_uuids, item.is_exercise)
        for item in result['statements']
    ] == [(['block-0'], True)]
    assert result['procedures'] == []
    assert role_typer.calls == []
    assert statement_partitioner.calls == []
    assert procedure_partitioner.calls == []
