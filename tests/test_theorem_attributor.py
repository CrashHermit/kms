"""Theorem attributor: the statement/proof split around the LLM passes.

The identity pass (label/number/title/field/proof_start) and the two bodylist passes are
injected via a scripted module, so these tests exercise the real split/assembly logic —
statement vs proof at proof_start, label peeling, proofs population — without dspy."""

import asyncio

from kms.core.state import ASTNode, BodySegment, Entity, EntityType, NodeType
from kms.entity.attributors.theorem import Identity, attribute_theorem


def _nodes():
    return [
        ASTNode(type=NodeType.HEADER, content="Theorem 3.2", id=0, seg_index=0),
        ASTNode(
            type=NodeType.PARAGRAPH,
            content="Let $n \\ge 3$. Then $Z(S_n)$ is trivial.",
            id=1,
            seg_index=0,
        ),
        ASTNode(
            type=NodeType.PARAGRAPH,
            content="Proof. Suppose $\\sigma \\in Z(S_n)$ ... hence trivial.",
            id=2,
            seg_index=0,
        ),
    ]


class _ScriptedModule:
    def __init__(self, identity, statement_bl=None, proof_bl=None):
        self._identity = identity
        self._statement_bl = statement_bl or []
        self._proof_bl = proof_bl or []

    async def identity(self, members):
        return self._identity

    async def statement_body(self, contents):
        return list(self._statement_bl)

    async def proof_body(self, contents):
        return list(self._proof_bl)


def _run(entity, nodes, module):
    return asyncio.run(attribute_theorem(entity, {n.id: n for n in nodes}, module=module))


def test_split_holds_out_proof_and_peels_label():
    nodes = _nodes()
    entity = Entity(type=EntityType.THEOREM, members=[0, 1, 2])
    ident = Identity(
        label="Theorem 3.2", number="3.2", title="Center is Trivial", field="algebra", proof_start=2
    )
    module = _ScriptedModule(
        ident,
        statement_bl=[BodySegment(description="Let $n \\ge 3$.", action="assumption")],
        proof_bl=[BodySegment(description="Suppose ...", action="deduction")],
    )
    e = _run(entity, nodes, module)

    assert e.label == "Theorem 3.2"
    assert e.number == "3.2"
    assert e.field == "algebra"
    # Statement = members before proof_start, label node dropped; proof held out of contents.
    assert e.contents == ["Let $n \\ge 3$. Then $Z(S_n)$ is trivial."]
    assert [s.action for s in e.bodylist] == ["assumption"]
    # The proof went into proofs, with its own contents + bodylist.
    assert len(e.proofs) == 1
    assert e.proofs[0].contents == ["Proof. Suppose $\\sigma \\in Z(S_n)$ ... hence trivial."]
    assert [s.action for s in e.proofs[0].bodylist] == ["deduction"]


def test_no_proof_leaves_proofs_empty():
    nodes = _nodes()[:2]  # label + statement only
    entity = Entity(type=EntityType.THEOREM, members=[0, 1])
    ident = Identity(label="Theorem 3.2", number="3.2", title="X", field="algebra", proof_start=-1)
    e = _run(entity, nodes, _ScriptedModule(ident))

    assert e.proofs == []
    assert e.contents == ["Let $n \\ge 3$. Then $Z(S_n)$ is trivial."]


def test_out_of_range_proof_start_is_treated_as_no_proof():
    nodes = _nodes()
    entity = Entity(type=EntityType.THEOREM, members=[0, 1, 2])
    ident = Identity(label="Theorem 3.2", number="3.2", title="X", field="algebra", proof_start=9)
    e = _run(entity, nodes, _ScriptedModule(ident))

    assert e.proofs == []
    # All non-label members stay in the statement contents.
    assert len(e.contents) == 2
