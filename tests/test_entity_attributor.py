"""Stage 2 attribution: role assignment, mechanical defaults, metadata lift, and fallback."""

import asyncio

from module.state import ASTNode, NodeType, EntityType, EntityRole, Entity, Member
from module.entity_attributor import EntityAttributorNode, Module


def _nodes():
    return [
        ASTNode(type=NodeType.PARAGRAPH, content="Thm", id=1, seg_index=0),
        ASTNode(type=NodeType.PARAGRAPH, content="pf1", id=2, seg_index=0),
        ASTNode(type=NodeType.PARAGRAPH, content="pf2", id=3, seg_index=0),
        ASTNode(type=NodeType.PARAGRAPH, content="Def", id=4, seg_index=0),
        ASTNode(type=NodeType.PROBLEM, content="7. do", id=5, seg_index=0, number="7", instruction="Find x"),
    ]


def _entities():
    return [
        Entity(type=EntityType.THEOREM, members=[Member(1), Member(2), Member(3)], id=0),
        Entity(type=EntityType.DEFINITION, members=[Member(4)], id=1),
        Entity(type=EntityType.PROBLEM, members=[Member(5)], id=2),
    ]


def test_dispatch_only_fans_out_multi_member_theorem_or_problem():
    at = EntityAttributorNode(module=object())
    sends = at.dispatch({"nodes": _nodes(), "entities": _entities()})
    assert [s.arg["entity_id"] for s in sends] == [0]  # def + single-member problem are mechanical


def test_collect_applies_roles_defaults_definitions_and_lifts_metadata():
    at = EntityAttributorNode(module=object())
    out = at.collect({
        "nodes": _nodes(),
        "entities": _entities(),
        "attribute_results": [(0, ["statement", "proof", "proof"])],
    })["entities"]
    assert [m.role.value for m in out[0].members] == ["statement", "proof", "proof"]
    assert [m.role.value for m in out[1].members] == ["statement"]  # definition mechanical
    assert out[2].members[0].role == EntityRole.STATEMENT
    assert out[2].number == "7" and out[2].instruction == "Find x"  # lifted from the node
    assert out[0].number is None  # theorem node carries no metadata


def _labeler(roles):
    """A Module whose LLM call returns `roles`, bypassing __init__/text_lm."""
    m = Module.__new__(Module)

    class _L:
        async def acall(self, **_):
            class _R:
                pass
            r = _R()
            r.roles = roles
            return r

    m.labeler = _L()
    return m


def test_aforward_validates_length_and_falls_back_positionally():
    run = lambda coro: asyncio.run(coro)
    # aligned + valid -> passthrough
    assert run(_labeler(["statement", "proof", "proof"]).aforward(EntityType.THEOREM, ["a", "b", "c"])) == \
        ["statement", "proof", "proof"]
    # wrong length -> positional fallback (first statement, rest secondary)
    assert run(_labeler(["statement"]).aforward(EntityType.THEOREM, ["a", "b", "c"])) == \
        ["statement", "proof", "proof"]
    # foreign role for the type -> per-slot positional fallback (solution invalid for theorem)
    assert run(_labeler(["statement", "solution", "proof"]).aforward(EntityType.THEOREM, ["a", "b", "c"])) == \
        ["statement", "proof", "proof"]
    # problem's secondary role is solution
    assert run(_labeler(["x", "y"]).aforward(EntityType.PROBLEM, ["a", "b"])) == ["statement", "solution"]
