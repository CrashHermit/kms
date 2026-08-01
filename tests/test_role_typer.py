"""Role typer: diagnoses group composition, creates Statement and Procedure
hubs."""

import asyncio

import pytest

from kms.core import models
from kms.ingestion import role_typer


class _ScriptedModule:
    def __init__(self, roles):
        self._roles = list(roles)

    async def acall(self, contents):
        return self._roles.pop(0)


def test_creates_a_hub_per_role():
    nodes = [
        models.ParagraphNode(content='Theorem 2.1', id=0),
        models.ParagraphNode(content='Proof. ...', id=1),
        models.ParagraphNode(content='Exercise 3', id=2),
    ]
    by_id = {node.id: node for node in nodes}
    module = _ScriptedModule([['statement'], ['procedure'], ['statement']])
    statements, procedures = asyncio.run(
        role_typer.type_roles([[0], [1], [2]], by_id, module)
    )
    assert [s.block for s in statements] == [[0], [2]]
    assert [s.members for s in statements] == [[0], [2]]
    assert [p.block for p in procedures] == [[1]]
    assert procedures[0].members == [1]


def test_a_both_block_creates_both_independent_hubs():
    # A block that both states and works out gets a statement hub AND a
    # procedure hub — separate vertices, sharing the same block.
    nodes = [
        models.ParagraphNode(content='Example 4.2. Compute ...', id=0),
        models.ParagraphNode(content='The value is 4.', id=1),
    ]
    by_id = {node.id: node for node in nodes}
    module = _ScriptedModule([['statement', 'procedure']])
    statements, procedures = asyncio.run(
        role_typer.type_roles([[0, 1]], by_id, module)
    )
    assert statements[0].block == [0, 1]
    assert statements[0].members == [0, 1]
    assert procedures[0].block == [0, 1]
    assert procedures[0].members == [0, 1]


def test_a_statement_cannot_be_built_without_a_block():
    # Required: a hub is only ever built from an already-found span, so a
    # block-less one is a bug that should stop here rather than travel on and
    # name nothing.
    with pytest.raises(TypeError):
        models.Statement()


def test_a_statement_is_not_an_ast_node():
    # The stream cannot hold one even by accident: Statement is a peer of
    # Procedure, not an ASTNode.
    nodes = [models.ParagraphNode(content='Theorem 2.1', id=0)]
    module = _ScriptedModule([['statement']])
    statements, _ = asyncio.run(
        role_typer.type_roles([[0]], {0: nodes[0]}, module)
    )
    assert not isinstance(statements[0], models.ASTNode)


def test_the_node_stream_is_left_alone():
    # The overlay is built beside `nodes`, not into it: a statement standing in
    # the stream would make every stage that walks nodes read the members
    # twice.
    nodes = [
        models.ParagraphNode(content='Theorem 2.1', id=0),
        models.ParagraphNode(content='Proof. ...', id=1),
    ]
    by_id = {node.id: node for node in nodes}
    module = _ScriptedModule([['statement']])
    asyncio.run(role_typer.type_roles([[0, 1]], by_id, module))
    assert [type(node) for node in nodes] == [
        models.ParagraphNode,
        models.ParagraphNode,
    ]
    assert by_id[0] is nodes[0]  # not swapped out for the statement


def test_an_unusable_role_falls_back_to_statement():
    nodes = [models.ParagraphNode(content='Theorem 2.1', id=0)]
    by_id = {0: nodes[0]}
    module = _ScriptedModule([['nonsense']])
    statements, procedures = asyncio.run(
        role_typer.type_roles([[0]], by_id, module)
    )
    assert [s.block for s in statements] == [[0]]
    assert procedures == []


def test_no_spans_is_a_noop():
    assert asyncio.run(role_typer.type_roles([], {})) == ([], [])


def test_node_run_writes_the_hub_channels():
    nodes = [
        models.ParagraphNode(content='Theorem 2.1', id=0),
        models.ParagraphNode(content='Proof. ...', id=1),
    ]
    node = role_typer.RoleTyperNode(
        module=_ScriptedModule([['statement'], ['procedure']])
    )
    out = asyncio.run(node.run({'nodes': nodes, 'spans': [[0], [1]]}))
    assert set(out) == {'statements', 'procedures'}  # `nodes` is not written
    statements = out['statements']
    procedures = out['procedures']
    assert [s.block for s in statements] == [[0]]
    assert [p.block for p in procedures] == [[1]]
    assert all(isinstance(node, models.ParagraphNode) for node in nodes)


def test_node_run_on_an_empty_spans_channel_is_a_noop():
    node = role_typer.RoleTyperNode(module=_ScriptedModule([]))
    out = asyncio.run(
        node.run(
            {'nodes': [models.ParagraphNode(content='x', id=0)], 'spans': []}
        )
    )
    assert out['statements'] == []
    assert out['procedures'] == []
