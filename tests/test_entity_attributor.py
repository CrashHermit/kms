"""Stage 2 attribution: role assignment, mechanical defaults, metadata lift, and fallback."""

import asyncio

from module.state import ASTNode, NodeType, EntityType, EntityRole, Entity, Member
from module.entity_attributor import EntityAttributorNode, Module, _is_marker


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


def test_is_marker_recognizes_proof_and_solution_headings():
    assert _is_marker("Solution") and _is_marker("### Solution") and _is_marker("**Proof**") and _is_marker("Proof:")
    assert not _is_marker("Solution: compute x")  # has a body -> not a bare marker
    assert not _is_marker("The proof follows") and not _is_marker(None)


def test_marker_split_is_deterministic_and_skips_the_llm():
    # Example / title / setup / "Solution" / body — the case the LLM labelled inconsistently.
    nodes = [
        ASTNode(type=NodeType.HEADER, content="Example 2.30", id=0, seg_index=0),
        ASTNode(type=NodeType.HEADER, content="Classifying a Discontinuity", id=1, seg_index=0),
        ASTNode(type=NodeType.PARAGRAPH, content="In Example 2.26 we showed ...", id=2, seg_index=0),
        ASTNode(type=NodeType.HEADER, content="### Solution", id=3, seg_index=0),
        ASTNode(type=NodeType.PARAGRAPH, content="To classify ...", id=4, seg_index=0),
    ]
    ent = Entity(type=EntityType.PROBLEM, members=[Member(i) for i in range(5)], id=0)
    at = EntityAttributorNode(module=object())
    # Marker present -> no LLM fan-out at all.
    assert at.dispatch({"nodes": nodes, "entities": [ent]}) == "entity_attributor_collect"
    # collect splits at the marker with no attribute_results in state.
    out = at.collect({"nodes": nodes, "entities": [ent]})["entities"][0]
    assert [m.role.value for m in out.members] == ["statement", "statement", "statement", "solution", "solution"]


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
