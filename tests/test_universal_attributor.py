"""Universal attributor: one identity pass fills label/number/title and the OPEN, induced `type`.
The pass is injected via a scripted module, so this exercises the pure logic — type normalization,
label peeling, contents assembly — without dspy."""

import asyncio

from kms.core import models
from kms.entity.attributors.universal import (
    Identity,
    UniversalAttributorNode,
    attribute,
)


def _nodes():
    return [
        models.ASTNode(
            type=models.NodeType.HEADER,
            content='Theorem 3.2',
            id=0,
            segment_index=0,
        ),
        models.ASTNode(
            type=models.NodeType.PARAGRAPH,
            content='The center of $S_n$ is trivial.',
            id=1,
            segment_index=0,
        ),
    ]


class _ScriptedModule:
    """A stand-in Module returning a fixed identity."""

    def __init__(self, identity):
        self._identity = identity

    async def identity(self, members):
        return self._identity


def _run(entity, nodes, identity):
    return asyncio.run(
        attribute(entity, {n.id: n for n in nodes}, _ScriptedModule(identity))
    )


def test_attributes_are_written_and_the_standalone_label_node_is_dropped():
    entity = models.Entity(members=[0, 1])
    e = _run(
        entity,
        _nodes(),
        Identity(
            label='Theorem 3.2',
            number='3.2',
            title='Center of Symmetric Group is Trivial',
            type='theorem',
        ),
    )
    assert (e.label, e.number, e.type) == ('Theorem 3.2', '3.2', 'theorem')
    assert e.contents == ['The center of $S_n$ is trivial.']


def test_the_type_is_open_a_physics_law_types_itself():
    # nothing validates the type against a math vocabulary — that is the whole point
    e = _run(
        models.Entity(members=[0, 1]),
        _nodes(),
        Identity(label='', number='', title='Ohm', type='law'),
    )
    assert e.type == 'law'


def test_the_worked_part_stays_in_contents_for_the_procedure_finder():
    # the attributor does NOT split statement from derivation; that is one pass later
    nodes = _nodes() + [
        models.ASTNode(
            type=models.NodeType.PARAGRAPH,
            content='Proof. Clear.',
            id=2,
            segment_index=0,
        )
    ]
    e = _run(
        models.Entity(members=[0, 1, 2]),
        nodes,
        Identity(label='Theorem 3.2', number='3.2', title='X', type='theorem'),
    )
    assert e.contents == ['The center of $S_n$ is trivial.', 'Proof. Clear.']


def test_a_fused_label_is_peeled_but_its_statement_kept():
    nodes = [
        models.ASTNode(
            type=models.NodeType.PARAGRAPH,
            content='Definition 1.2. A group is a set with an operation.',
            id=0,
            segment_index=0,
        )
    ]
    e = _run(
        models.Entity(members=[0]),
        nodes,
        Identity(
            label='Definition 1.2',
            number='1.2',
            title='Group',
            type='definition',
        ),
    )
    assert e.contents == ['A group is a set with an operation.']


def test_node_run_writes_back_to_the_block_channel():
    node = UniversalAttributorNode(
        module=_ScriptedModule(
            Identity(label='Example 1', number='1', title='X', type='example')
        )
    )
    out = asyncio.run(
        node.run(
            {
                'nodes': _nodes(),
                'block_entities': [models.Entity(members=[0, 1])],
            }
        )
    )
    assert list(out) == ['block_entities']
    assert out['block_entities'][0].type == 'example'
