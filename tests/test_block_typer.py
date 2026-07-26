"""Block typer: the open `type` induction over the role-typed block overlay. Covers the
open-vocabulary normalisation (spelling only, never validation against a list) and the
graph-node wrapper. The LLM call is stubbed — this tests everything around it."""

import asyncio

from kms.core import models
from kms.entity import block_typer


def _nodes(*contents):
    return [
        models.ASTNode(
            type=models.NodeType.PARAGRAPH, content=text, id=i, segment_index=0
        )
        for i, text in enumerate(contents)
    ]


class _ScriptedModule:
    """A stand-in Module returning pre-scripted types, recording the members it saw."""

    def __init__(self, types):
        self._types = list(types)
        self.seen = []

    async def block_type(self, members):
        self.seen.append(members)
        return self._types.pop(0)


# --- normalisation: spelling only, open vocabulary ---


def test_type_is_normalized_but_not_validated_against_a_list():
    # a non-math genre passes straight through — the vocabulary is open
    assert block_typer.normalize_type('  Second   LAW ') == 'second law'
    assert block_typer.normalize_type('Mechanism') == 'mechanism'
    assert block_typer.normalize_type(None) == ''


# --- the typing pass ---


def test_writes_the_induced_type_onto_each_block():
    nodes = _nodes('Theorem 2.1', 'Exercise 3')
    by_id = {node.id: node for node in nodes}
    entities = [models.Entity(members=[0]), models.Entity(members=[1])]
    module = _ScriptedModule(['theorem', 'exercise'])
    asyncio.run(block_typer.type_blocks(entities, by_id, module))
    assert [entity.type for entity in entities] == ['theorem', 'exercise']


def test_the_typer_sees_the_blocks_member_nodes():
    nodes = _nodes('Theorem 2.1', 'Every group has an identity.')
    by_id = {node.id: node for node in nodes}
    entities = [models.Entity(members=[0, 1])]
    module = _ScriptedModule(['theorem'])
    asyncio.run(block_typer.type_blocks(entities, by_id, module))
    assert [node.id for node in module.seen[0]] == [0, 1]


def test_members_of_skips_ids_missing_from_the_stream():
    nodes = _nodes('kept')
    entity = models.Entity(members=[0, 99])
    assert block_typer.members_of(entity, {0: nodes[0]}) == [nodes[0]]


def test_an_empty_overlay_is_a_noop():
    assert asyncio.run(block_typer.type_blocks([], {})) == []


# --- graph node ---


def test_node_run_writes_the_entities_channel():
    nodes = _nodes('Definition 1.1')
    entity = models.Entity(members=[0])
    node = block_typer.BlockTyperNode(module=_ScriptedModule(['definition']))
    out = asyncio.run(node.run({'nodes': nodes, 'entities': [entity]}))
    assert list(out) == ['entities']
    assert out['entities'][0].type == 'definition'


def test_node_run_on_an_empty_overlay_is_a_noop():
    node = block_typer.BlockTyperNode(module=_ScriptedModule([]))
    assert asyncio.run(node.run({'nodes': [], 'entities': []})) == {
        'entities': []
    }
