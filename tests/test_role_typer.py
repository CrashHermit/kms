"""Role typer: diagnoses group composition, creates StatementNodes + Procedures."""

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
    sids, pids = asyncio.run(
        role_typer.type_roles([[0], [1], [2]], by_id, module)
    )
    assert sids == [0, 1, 2]  # every span gets a StatementNode
    assert pids == [1]         # only the procedure span
    assert isinstance(by_id[0], models.StatementNode)
    assert by_id[0].statement_of == [0]
    assert by_id[0].procedures == []
    assert isinstance(by_id[1], models.StatementNode)
    assert by_id[1].statement_of == [1]
    assert len(by_id[1].procedures) == 1
    assert isinstance(by_id[2], models.StatementNode)
    assert by_id[2].statement_of == [2]
    assert by_id[2].procedures == []


def test_an_unusable_role_falls_back_to_statement():
    nodes = [models.ParagraphNode(content='Theorem 2.1', id=0)]
    by_id = {0: nodes[0]}
    module = _ScriptedModule(['nonsense'])
    sids, pids = asyncio.run(
        role_typer.type_roles([[0]], by_id, module)
    )
    assert sids == [0]
    assert pids == []
    assert isinstance(by_id[0], models.StatementNode)


def test_no_spans_is_a_noop():
    assert asyncio.run(role_typer.type_roles([], {})) == ([], [])


def test_node_run_writes_both_channels():
    nodes = [
        models.ParagraphNode(content='Theorem 2.1', id=0),
        models.ParagraphNode(content='Proof. ...', id=1),
    ]
    node = role_typer.RoleTyperNode(
        module=_ScriptedModule(['statement', 'procedure'])
    )
    out = asyncio.run(node.run({'nodes': nodes, 'spans': [[0], [1]]}))
    assert out['statement_ids'] == [0, 1]
    assert out['procedure_ids'] == [1]
    assert isinstance(nodes[0], models.StatementNode)
    assert nodes[0].procedures == []
    assert isinstance(nodes[1], models.StatementNode)
    assert len(nodes[1].procedures) == 1


def test_node_run_on_an_empty_spans_channel_is_a_noop():
    node = role_typer.RoleTyperNode(module=_ScriptedModule([]))
    out = asyncio.run(
        node.run({'nodes': [models.ParagraphNode(content='x', id=0)], 'spans': []})
    )
    assert out['statement_ids'] == []
    assert out['procedure_ids'] == []
