"""Role typer: diagnoses group composition, creates StatementNodes +
Procedures."""

import asyncio

from kms.core import models
from kms.ingestion import role_typer


class _ScriptedModule:
    def __init__(self, roles):
        self._roles = list(roles)

    async def acall(self, contents):
        return self._roles.pop(0)


def test_splits_statements_from_procedures():
    nodes = [
        models.ParagraphNode(content='Theorem 2.1', id=0),
        models.ParagraphNode(content='Proof. ...', id=1),
        models.ParagraphNode(content='Exercise 3', id=2),
    ]
    by_id = {node.id: node for node in nodes}
    module = _ScriptedModule(['statement', 'procedure', 'statement'])
    statements = asyncio.run(
        role_typer.type_roles([[0], [1], [2]], by_id, module)
    )
    assert [s.id for s in statements] == [0, 1, 2]  # every span gets one
    assert [s.statement_of for s in statements] == [[0], [1], [2]]
    assert statements[0].procedures == []
    assert len(statements[1].procedures) == 1  # only the procedure span
    assert statements[2].procedures == []


def test_the_node_stream_is_left_alone():
    # The overlay is built beside `nodes`, not into it: a statement standing in
    # the stream would carry its whole group's text and make every stage that
    # walks nodes read the members twice.
    nodes = [
        models.ParagraphNode(content='Theorem 2.1', id=0),
        models.ParagraphNode(content='Proof. ...', id=1),
    ]
    by_id = {node.id: node for node in nodes}
    module = _ScriptedModule(['statement'])
    asyncio.run(role_typer.type_roles([[0, 1]], by_id, module))
    assert [type(node) for node in nodes] == [
        models.ParagraphNode,
        models.ParagraphNode,
    ]
    assert by_id[0] is nodes[0]  # not swapped out for the statement


def test_an_unusable_role_falls_back_to_statement():
    nodes = [models.ParagraphNode(content='Theorem 2.1', id=0)]
    by_id = {0: nodes[0]}
    module = _ScriptedModule(['nonsense'])
    statements = asyncio.run(role_typer.type_roles([[0]], by_id, module))
    assert [s.id for s in statements] == [0]
    assert statements[0].procedures == []


def test_no_spans_is_a_noop():
    assert asyncio.run(role_typer.type_roles([], {})) == []


def test_node_run_writes_the_statements_channel():
    nodes = [
        models.ParagraphNode(content='Theorem 2.1', id=0),
        models.ParagraphNode(content='Proof. ...', id=1),
    ]
    node = role_typer.RoleTyperNode(
        module=_ScriptedModule(['statement', 'procedure'])
    )
    out = asyncio.run(node.run({'nodes': nodes, 'spans': [[0], [1]]}))
    assert set(out) == {'statements'}  # `nodes` is not written back
    statements = out['statements']
    assert [s.id for s in statements] == [0, 1]
    assert statements[0].procedures == []
    assert len(statements[1].procedures) == 1
    assert all(isinstance(node, models.ParagraphNode) for node in nodes)


def test_node_run_on_an_empty_spans_channel_is_a_noop():
    node = role_typer.RoleTyperNode(module=_ScriptedModule([]))
    out = asyncio.run(
        node.run(
            {'nodes': [models.ParagraphNode(content='x', id=0)], 'spans': []}
        )
    )
    assert out['statements'] == []
