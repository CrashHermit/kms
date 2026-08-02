"""Hub builder: role typing + both-block partitioning, one pass. Pure logic
with scripted modules."""

import asyncio

import pytest

from kms.core import models
from kms.ingestion import hub_builder

# --- Role typer tests ------------------------------------------------------


class _ScriptedRoles:
    def __init__(self, roles):
        self._roles = list(roles)

    async def acall(self, contents):
        return self._roles.pop(0)


def test_creates_a_hub_per_role():
    nodes = [
        models.ASTNode(type='paragraph', content='Theorem 2.1', id=0),
        models.ASTNode(type='paragraph', content='Proof. ...', id=1),
        models.ASTNode(type='paragraph', content='Exercise 3', id=2),
    ]
    by_id = {node.id: node for node in nodes}
    module = _ScriptedRoles([(True, False), (False, True), (True, False)])
    statements, procedures = asyncio.run(
        hub_builder.build_hubs([[0], [1], [2]], by_id, role_module=module)
    )
    assert [s.block for s in statements] == [[0], [2]]
    assert [s.members for s in statements] == [[0], [2]]
    assert [p.block for p in procedures] == [[1]]
    assert procedures[0].members == [1]


def test_a_both_block_creates_both_independent_hubs():
    nodes = [
        models.ASTNode(type='paragraph', content='Example 4.2. Compute ...', id=0),
        models.ASTNode(type='paragraph', content='The value is 4.', id=1),
    ]
    by_id = {node.id: node for node in nodes}
    role_mod = _ScriptedRoles([(True, True)])
    # Both roles present → partitioners run. Script them to keep full span.
    stmt_mod = _ScriptedPositions([[0, 1]])
    proc_mod = _ScriptedPositions([[0, 1]])
    statements, procedures = asyncio.run(
        hub_builder.build_hubs(
            [[0, 1]],
            by_id,
            role_module=role_mod,
            statement_partitioner=stmt_mod,
            procedure_partitioner=proc_mod,
        )
    )
    assert statements[0].block == [0, 1]
    assert statements[0].members == [0, 1]
    assert procedures[0].block == [0, 1]
    assert procedures[0].members == [0, 1]


def test_a_statement_cannot_be_built_without_a_block():
    with pytest.raises(TypeError):
        models.Statement()


def test_a_statement_is_not_an_ast_node():
    nodes = [models.ASTNode(type='paragraph', content='Theorem 2.1', id=0)]
    module = _ScriptedRoles([(True, False)])
    statements, _ = asyncio.run(
        hub_builder.build_hubs([[0]], {0: nodes[0]}, role_module=module)
    )
    assert not isinstance(statements[0], models.ASTNode)


def test_the_node_stream_is_left_alone():
    nodes = [
        models.ASTNode(type='paragraph', content='Theorem 2.1', id=0),
        models.ASTNode(type='paragraph', content='Proof. ...', id=1),
    ]
    by_id = {node.id: node for node in nodes}
    module = _ScriptedRoles([(True, False)])
    asyncio.run(hub_builder.build_hubs([[0, 1]], by_id, role_module=module))
    assert [node.type for node in nodes] == ['paragraph', 'paragraph']
    assert by_id[0] is nodes[0]


def test_no_spans_is_a_noop():
    assert asyncio.run(
        hub_builder.build_hubs(
            [],
            {},
            role_module=_ScriptedRoles([]),
        )
    ) == ([], [])


def test_node_run_writes_the_hub_channels():
    nodes = [
        models.ASTNode(type='paragraph', content='Theorem 2.1', id=0),
        models.ASTNode(type='paragraph', content='Proof. ...', id=1),
    ]
    node = hub_builder.HubBuilderNode(
        role_module=_ScriptedRoles([(True, False), (False, True)])
    )
    out = asyncio.run(node.run({'nodes': nodes, 'spans': [[0], [1]]}))
    assert set(out) == {'statements', 'procedures'}
    statements = out['statements']
    procedures = out['procedures']
    assert [s.block for s in statements] == [[0]]
    assert [p.block for p in procedures] == [[1]]
    assert all(n.type == 'paragraph' for n in nodes)


def test_node_run_on_an_empty_spans_channel_is_a_noop():
    node = hub_builder.HubBuilderNode(role_module=_ScriptedRoles([]))
    out = asyncio.run(
        node.run(
            {'nodes': [models.ASTNode(type='paragraph', content='x', id=0)], 'spans': []}
        )
    )
    assert out['statements'] == []
    assert out['procedures'] == []


# --- Partitioner tests -----------------------------------------------------


class _ScriptedPositions:
    def __init__(self, positions):
        self._positions = list(positions)

    async def aforward(self, current_nodes):
        return self._positions.pop(0)


def _nodes():
    return {
        0: models.ASTNode(type='paragraph', content='Example 4.2. Compute ...', id=0),
        1: models.ASTNode(type='paragraph', content='Integrate ...', id=1),
        2: models.ASTNode(type='paragraph', content='Hence the value is 4.', id=2),
    }


def _both_modules(stmt_positions, proc_positions):
    return (
        _ScriptedRoles([(True, True)]),
        _ScriptedPositions(stmt_positions),
        _ScriptedPositions(proc_positions),
    )


def test_both_block_partitions_statement_members():
    role_mod, stmt_mod, proc_mod = _both_modules([[0]], [[]])
    statements, procedures = asyncio.run(
        hub_builder.build_hubs(
            [[0, 1, 2]],
            _nodes(),
            role_module=role_mod,
            statement_partitioner=stmt_mod,
            procedure_partitioner=proc_mod,
        )
    )
    assert statements[0].members == [0]
    assert statements[0].block == [0, 1, 2]


def test_both_block_partitions_procedure_members():
    role_mod, stmt_mod, proc_mod = _both_modules([[]], [[1, 2]])
    statements, procedures = asyncio.run(
        hub_builder.build_hubs(
            [[0, 1, 2]],
            _nodes(),
            role_module=role_mod,
            statement_partitioner=stmt_mod,
            procedure_partitioner=proc_mod,
        )
    )
    assert procedures[0].members == [1, 2]
    assert procedures[0].block == [0, 1, 2]


def test_single_role_statement_skips_partitioning():
    nodes = _nodes()
    # Single role: no procedure, partitioners would fail if called.
    role_mod = _ScriptedRoles([(True, False)])
    statements, procedures = asyncio.run(
        hub_builder.build_hubs(
            [[0, 1, 2]],
            nodes,
            role_module=role_mod,
        )
    )
    assert statements[0].members == [0, 1, 2]
    assert procedures == []


def test_single_role_procedure_skips_partitioning():
    nodes = _nodes()
    role_mod = _ScriptedRoles([(False, True)])
    statements, procedures = asyncio.run(
        hub_builder.build_hubs(
            [[0, 1, 2]],
            nodes,
            role_module=role_mod,
        )
    )
    assert statements == []
    assert procedures[0].members == [0, 1, 2]


def test_an_empty_selection_keeps_the_full_block():
    role_mod, stmt_mod, proc_mod = _both_modules([[]], [[]])
    statements, procedures = asyncio.run(
        hub_builder.build_hubs(
            [[0, 1, 2]],
            _nodes(),
            role_module=role_mod,
            statement_partitioner=stmt_mod,
            procedure_partitioner=proc_mod,
        )
    )
    assert statements[0].members == [0, 1, 2]
    assert procedures[0].members == [0, 1, 2]


def test_out_of_range_positions_are_dropped():
    role_mod, stmt_mod, proc_mod = _both_modules([[0, 99]], [[]])
    statements, _ = asyncio.run(
        hub_builder.build_hubs(
            [[0, 1, 2]],
            _nodes(),
            role_module=role_mod,
            statement_partitioner=stmt_mod,
            procedure_partitioner=proc_mod,
        )
    )
    assert statements[0].members == [0]
