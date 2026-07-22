"""Definition attributor: deterministic content assembly around the two LLM passes.

The identity pass (label/number/title/field) and the bodylist pass are injected via a
scripted module, so these tests exercise the real assembly logic — peeling the label off
the content (dropping a pure-label node, keeping a fused one) — without dspy or network."""

import asyncio

from module.state import ASTNode, NodeType, Entity, EntityType, BodySegment
from module.definition_attributor import attribute_definition, Identity


def _nodes():
    return [
        ASTNode(type=NodeType.HEADER, content="1.2 Definition", id=0, seg_index=0),
        ASTNode(type=NodeType.PARAGRAPH, content="A vector space is a set $V$ ...", id=1, seg_index=0),
        ASTNode(type=NodeType.MATH, content="$$V \\times V \\to V$$", id=2, seg_index=0),
    ]


class _ScriptedModule:
    def __init__(self, identity: Identity, bodylist):
        self._identity, self._bodylist = identity, bodylist

    async def identity(self, members):
        return self._identity

    async def body(self, contents):
        return list(self._bodylist)


def _run(entity, nodes, module):
    by_id = {n.id: n for n in nodes}
    return asyncio.run(attribute_definition(entity, by_id, module=module))


def test_flagged_label_node_is_dropped_from_contents():
    nodes = _nodes()
    entity = Entity(type=EntityType.DEFINITION, members=[0, 1, 2])
    ident = Identity(label="1.2 Definition", number="1.2", title="Vector Space", field="algebra")
    module = _ScriptedModule(ident, [BodySegment(description="A vector space is a set $V$ ...", action="definition")])
    attrs = _run(entity, nodes, module)

    assert attrs.label == "1.2 Definition"
    assert attrs.number == "1.2"
    assert attrs.title == "Vector Space"
    assert attrs.field == "algebra"
    # The pure-label node ("1.2 Definition") strips to empty and is dropped; statement + math remain.
    assert attrs.contents == ["A vector space is a set $V$ ...", "$$V \\times V \\to V$$"]
    assert [s.action for s in attrs.bodylist] == ["definition"]


def test_fused_label_is_stripped_from_contents():
    node = ASTNode(type=NodeType.PARAGRAPH, content="Definition 1.2 A group is a set with ...", id=5, seg_index=0)
    entity = Entity(type=EntityType.DEFINITION, members=[5])
    # Fused label: the prefix is peeled off the first content string, the node is kept.
    ident = Identity(label="Definition 1.2", number="1.2", title="Group", field="algebra")
    attrs = _run(entity, [node], _ScriptedModule(ident, []))

    assert attrs.number == "1.2"
    assert attrs.contents == ["A group is a set with ..."]


def test_no_label_leaves_number_none_and_keeps_members():
    node = ASTNode(type=NodeType.PARAGRAPH, content="A ring is a set with two operations ...", id=7, seg_index=0)
    entity = Entity(type=EntityType.DEFINITION, members=[7])
    ident = Identity(label=None, number=None, title="Ring", field="algebra")
    attrs = _run(entity, [node], _ScriptedModule(ident, []))

    assert attrs.label is None
    assert attrs.number is None
    assert attrs.contents == ["A ring is a set with two operations ..."]
