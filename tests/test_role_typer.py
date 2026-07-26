"""Role typer: the block/derivation split over the finder's untyped spans. Covers the
closed-vocabulary fallback, document-order preservation, and the graph-node wrapper. The LLM
call is stubbed — this tests everything around it."""

import asyncio

from kms.core import models
from kms.entity import role_typer


def _nodes(*contents):
    return [
        models.ASTNode(
            type=models.NodeType.PARAGRAPH, content=text, id=i, segment_index=0
        )
        for i, text in enumerate(contents)
    ]


class _ScriptedModule:
    """A stand-in Module returning pre-scripted roles, recording what it was asked."""

    def __init__(self, roles):
        self._roles = list(roles)
        self.seen = []

    async def role(self, contents):
        self.seen.append(contents)
        return self._roles.pop(0)


# --- contents assembly (deterministic, no LLM) ---


def test_contents_joins_members_in_order():
    nodes = _nodes('Theorem 2.1', 'Every group has an identity.')
    by_id = {node.id: node for node in nodes}
    assert role_typer.contents_of([0, 1], by_id) == (
        'Theorem 2.1\n\nEvery group has an identity.'
    )


def test_contents_skips_unknown_ids_and_blank_nodes():
    nodes = _nodes('kept', '   ')
    by_id = {node.id: node for node in nodes}
    assert role_typer.contents_of([0, 1, 99], by_id) == 'kept'


# --- the split ---


def test_splits_blocks_from_derivations_preserving_order():
    nodes = _nodes('Theorem 2.1', 'Proof. ...', 'Exercise 3')
    by_id = {node.id: node for node in nodes}
    module = _ScriptedModule(['entity', 'procedure', 'entity'])
    entities, procedures = asyncio.run(
        role_typer.type_roles([[0], [1], [2]], by_id, module)
    )
    assert [entity.members for entity in entities] == [[0], [2]]
    assert procedures == [[1]]


def test_an_unusable_role_falls_back_to_entity():
    # A block wrongly demoted to a derivation would be hidden under its neighbour, so the
    # fallback is deliberately the block side.
    nodes = _nodes('Theorem 2.1')
    module = _ScriptedModule(['nonsense'])
    entities, procedures = asyncio.run(
        role_typer.type_roles([[0]], {0: nodes[0]}, module)
    )
    assert [entity.members for entity in entities] == [[0]]
    assert procedures == []


class _FakeClassify:
    """Stands in for the dspy predictor so the real Module.role normalisation is exercised."""

    def __init__(self, raw):
        self._raw = raw

    async def acall(self, contents):
        return type('R', (), {'role': self._raw})()


def _module_returning(raw):
    module = role_typer.Module.__new__(role_typer.Module)
    module.classify = _FakeClassify(raw)
    return module


def test_role_matching_is_case_and_whitespace_insensitive():
    assert asyncio.run(_module_returning(' PROCEDURE ').role('x')) == (
        'procedure'
    )


def test_module_falls_back_to_entity_for_an_unusable_answer():
    assert asyncio.run(_module_returning('nonsense').role('x')) == 'entity'
    assert asyncio.run(_module_returning('').role('x')) == 'entity'


def test_entities_carry_members_only():
    nodes = _nodes('Theorem 2.1')
    module = _ScriptedModule(['entity'])
    entities, _ = asyncio.run(
        role_typer.type_roles([[0]], {0: nodes[0]}, module)
    )
    assert entities[0].type is None  # block_typer fills this
    assert entities[0].label is None and entities[0].contents == []


def test_no_spans_is_a_noop():
    assert asyncio.run(role_typer.type_roles([], {})) == ([], [])


# --- graph node ---


def test_node_run_writes_both_channels():
    nodes = _nodes('Theorem 2.1', 'Proof. ...')
    node = role_typer.RoleTyperNode(
        module=_ScriptedModule(['entity', 'procedure'])
    )
    out = asyncio.run(node.run({'nodes': nodes, 'spans': [[0], [1]]}))
    assert set(out) == {'entities', 'procedure_spans'}
    assert [entity.members for entity in out['entities']] == [[0]]
    assert out['procedure_spans'] == [[1]]


def test_node_run_on_an_empty_spans_channel_is_a_noop():
    node = role_typer.RoleTyperNode(module=_ScriptedModule([]))
    assert asyncio.run(node.run({'nodes': [], 'spans': []})) == {
        'entities': [],
        'procedure_spans': [],
    }
