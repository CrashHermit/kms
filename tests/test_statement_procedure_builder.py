import asyncio

import pytest

from kms.construction import statement_procedure_builder
from kms.core import identity, models


class _ScriptedRoles:
    def __init__(self, roles):
        self._roles = list(roles)

    async def acall(self, current_nodes):
        return self._roles.pop(0)
class _RecordingRoles:
    def __init__(self):
        self.calls = []

    async def acall(self, **kwargs):
        self.calls.append(kwargs)
        return (True, True)


class _ScriptedPositions:
    def __init__(self, positions):
        self._positions = list(positions)

    async def aforward(self, current_nodes):
        return self._positions.pop(0)


def _nodes():
    return {
        0: models.SourceNode(
            type='paragraph', content='Example 4.2. Compute ...', uuid='node-0'
        ),
        1: models.SourceNode(type='paragraph', content='Integrate ...', uuid='node-1'),
        2: models.SourceNode(type='paragraph', content='Hence the value is 4.', uuid='node-2'),
    }


def _both_modules(stmt_positions, proc_positions):
    return (
        _ScriptedRoles([(True, True)]),
        _ScriptedPositions(stmt_positions),
        _ScriptedPositions(proc_positions),
    )


def test_assigns_statement_ids_and_dual_role_links():
    nodes = {
        0: models.SourceNode(type='paragraph', content='Example', uuid='node-0'),
        1: models.SourceNode(type='paragraph', content='Solution', uuid='node-1'),
    }
    role_mod = _ScriptedRoles([(True, True)])
    stmt_mod = _ScriptedPositions([[0]])
    proc_mod = _ScriptedPositions([[1]])

    statements, procedures = asyncio.run(
        statement_procedure_builder.build_statement_procedure_hubs(
            [[0, 1]],
            nodes,
            role_module=role_mod,
            statement_partitioner=stmt_mod,
            procedure_partitioner=proc_mod,
            source='book.pdf',
        )
    )

    expected = identity.statement_uuid('book.pdf', [0, 1])
    assert statements[0].uuid == expected
    assert procedures[0].statement_uuid == expected


def test_creates_a_hub_per_role():
    nodes = [
        models.SourceNode(type='paragraph', content='Theorem 2.1', uuid='node-0'),
        models.SourceNode(type='paragraph', content='Proof. ...', uuid='node-1'),
        models.SourceNode(type='paragraph', content='Exercise 3', uuid='node-2'),
    ]
    by_id = {i: node for i, node in enumerate(nodes)}
    module = _ScriptedRoles([(True, False), (False, True), (True, False)])
    statements, procedures = asyncio.run(
        statement_procedure_builder.build_statement_procedure_hubs(
            [[0], [1], [2]], by_id, role_module=module
        )
    )
    assert [s.block for s in statements] == [[0], [2]]
    assert [s.member_positions for s in statements] == [[0], [2]]
    assert [p.block for p in procedures] == [[1]]
    assert procedures[0].member_positions == [1]


def test_neither_role_is_skipped():
    nodes = [
        models.SourceNode(type='header', content='Learning Objectives', uuid='node-0'),
        models.SourceNode(type='paragraph', content='Theorem 2.1', uuid='node-1'),
    ]
    module = _ScriptedRoles([(False, False), (True, False)])
    statements, procedures = asyncio.run(
        statement_procedure_builder.build_statement_procedure_hubs(
            [[0], [1]],
            {i: node for i, node in enumerate(nodes)},
            role_module=module,
        )
    )
    assert [statement.block for statement in statements] == [[1]]
    assert procedures == []


def test_a_both_block_creates_both_independent_hubs():
    nodes = [
        models.SourceNode(type='paragraph', content='Example 4.2. Compute ...', uuid='node-0'),
        models.SourceNode(type='paragraph', content='The value is 4.', uuid='node-1'),
    ]
    by_id = {i: node for i, node in enumerate(nodes)}
    role_mod = _ScriptedRoles([(True, True)])
    stmt_mod = _ScriptedPositions([[0, 1]])
    proc_mod = _ScriptedPositions([[0, 1]])
    statements, procedures = asyncio.run(
        statement_procedure_builder.build_statement_procedure_hubs(
            [[0, 1]],
            by_id,
            role_module=role_mod,
            statement_partitioner=stmt_mod,
            procedure_partitioner=proc_mod,
        )
    )
    assert statements[0].block == [0, 1]
    assert statements[0].member_positions == [0, 1]
    assert procedures[0].block == [0, 1]
    assert procedures[0].member_positions == [0, 1]


def test_a_statement_cannot_be_built_without_a_block():
    with pytest.raises(TypeError):
        models.Statement()


def test_a_statement_is_not_an_ast_node():
    nodes = [models.SourceNode(type='paragraph', content='Theorem 2.1', uuid='node-0')]
    module = _ScriptedRoles([(True, False)])
    statements, _ = asyncio.run(
        statement_procedure_builder.build_statement_procedure_hubs(
            [[0]], {0: nodes[0]}, role_module=module
        )
    )
    assert not isinstance(statements[0], models.SourceNode)


def test_the_node_stream_is_left_alone():
    nodes = [
        models.SourceNode(type='paragraph', content='Theorem 2.1', uuid='node-0'),
        models.SourceNode(type='paragraph', content='Proof. ...', uuid='node-1'),
    ]
    by_id = {i: node for i, node in enumerate(nodes)}
    module = _ScriptedRoles([(True, False)])
    asyncio.run(
        statement_procedure_builder.build_statement_procedure_hubs(
            [[0, 1]], by_id, role_module=module
        )
    )
    assert [node.type for node in nodes] == ['paragraph', 'paragraph']
    assert by_id[0] is nodes[0]


def test_no_spans_is_a_noop():
    assert asyncio.run(
        statement_procedure_builder.build_statement_procedure_hubs(
            [],
            {},
            role_module=_ScriptedRoles([]),
        )
    ) == ([], [])
def test_undescribed_image_span_skips_role_typing():
    nodes = [
        models.SourceNode(
            type='image',
            content=None,
            assets=[models.VisualAsset(path='figure.png')],
            uuid='node-0',
        )
    ]
    role_module = _RecordingRoles()

    statements, procedures = asyncio.run(
        statement_procedure_builder.build_statement_procedure_hubs(
            [[0]], nodes, role_module=role_module
        )
    )

    assert statements == []
    assert procedures == []
    assert role_module.calls == []
    assert nodes[0].assets[0].path == 'figure.png'


def test_node_run_writes_the_hub_channels():
    nodes = [
        models.SourceNode(type='paragraph', content='Theorem 2.1', uuid='node-0'),
        models.SourceNode(type='paragraph', content='Proof. ...', uuid='node-1'),
    ]
    node = statement_procedure_builder.StatementProcedureBuilderNode(
        role_module=_ScriptedRoles([(True, False), (False, True)])
    )
    out = asyncio.run(
        node.run(
            {
                'nodes': nodes,
                'spans': [[0], [1]],
                'source': models.Source(key='book.pdf'),
            }
        )
    )
    assert set(out) == {'statements', 'procedures'}
    statements = out['statements']
    procedures = out['procedures']
    assert [s.block for s in statements] == [[0]]
    assert [p.block for p in procedures] == [[1]]
    assert all(n.type == 'paragraph' for n in nodes)
    assert statements[0].uuid == identity.statement_uuid(
        'book.pdf', [0]
    )


def test_node_run_on_an_empty_spans_channel_is_a_noop():
    node = statement_procedure_builder.StatementProcedureBuilderNode(
        role_module=_ScriptedRoles([])
    )
    out = asyncio.run(
        node.run(
            {
                'nodes': [models.SourceNode(type='paragraph', content='x', uuid='node-0')],
                'spans': [],
            }
        )
    )
    assert out['statements'] == []
    assert out['procedures'] == []


def test_both_block_partitions_statement_members():
    role_mod, stmt_mod, proc_mod = _both_modules([[0]], [[]])
    statements, procedures = asyncio.run(
        statement_procedure_builder.build_statement_procedure_hubs(
            [[0, 1, 2]],
            _nodes(),
            role_module=role_mod,
            statement_partitioner=stmt_mod,
            procedure_partitioner=proc_mod,
        )
    )
    assert statements[0].member_positions == [0]
    assert statements[0].block == [0, 1, 2]


def test_both_block_partitions_procedure_members():
    role_mod, stmt_mod, proc_mod = _both_modules([[]], [[1, 2]])
    statements, procedures = asyncio.run(
        statement_procedure_builder.build_statement_procedure_hubs(
            [[0, 1, 2]],
            _nodes(),
            role_module=role_mod,
            statement_partitioner=stmt_mod,
            procedure_partitioner=proc_mod,
        )
    )
    assert procedures[0].member_positions == [1, 2]
    assert procedures[0].block == [0, 1, 2]


def test_single_role_statement_skips_partitioning():
    nodes = _nodes()
    role_mod = _ScriptedRoles([(True, False)])
    statements, procedures = asyncio.run(
        statement_procedure_builder.build_statement_procedure_hubs(
            [[0, 1, 2]],
            nodes,
            role_module=role_mod,
        )
    )
    assert statements[0].member_positions == [0, 1, 2]
    assert procedures == []


def test_single_role_procedure_skips_partitioning():
    nodes = _nodes()
    role_mod = _ScriptedRoles([(False, True)])
    statements, procedures = asyncio.run(
        statement_procedure_builder.build_statement_procedure_hubs(
            [[0, 1, 2]],
            nodes,
            role_module=role_mod,
        )
    )
    assert statements == []
    assert procedures[0].member_positions == [0, 1, 2]


def test_an_empty_selection_is_applied_as_empty():
    role_mod, stmt_mod, proc_mod = _both_modules([[]], [[]])
    statements, procedures = asyncio.run(
        statement_procedure_builder.build_statement_procedure_hubs(
            [[0, 1, 2]],
            _nodes(),
            role_module=role_mod,
            statement_partitioner=stmt_mod,
            procedure_partitioner=proc_mod,
        )
    )
    assert statements[0].member_positions == []
    assert procedures[0].member_positions == []


def test_out_of_range_positions_fail_instead_of_being_dropped():
    role_mod, stmt_mod, proc_mod = _both_modules([[0, 99]], [[]])
    with pytest.raises(ValueError, match='partitioner position'):
        asyncio.run(
            statement_procedure_builder.build_statement_procedure_hubs(
                [[0, 1, 2]],
                _nodes(),
                role_module=role_mod,
                statement_partitioner=stmt_mod,
                procedure_partitioner=proc_mod,
            )
        )