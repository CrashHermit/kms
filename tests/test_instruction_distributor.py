"""Instruction distributor: stamp a lead-in's shared directive onto the Problems it governs,
by numeric range. The LLM is injected via a scripted module."""

import asyncio

from module.state import ASTNode, NodeType, EntityType, Entity
from module.instruction_distributor import (
    distribute_instructions,
    InstructionDistributorNode,
    _parse_number,
)


class _ScriptedRanger:
    """Replays one (start, end, instruction) per lead-in, keyed by a substring of its text."""

    def __init__(self, by_substring):
        self._by = dict(by_substring)

    async def range_of(self, lead_in: str):
        for key, triple in self._by.items():
            if key in lead_in:
                return triple
        return "", "", ""


def _nodes():
    # 0: lead-in (tagged) | 1-3: the three governed exercises | 4: a later ungoverned exercise
    return [
        ASTNode(type=NodeType.PARAGRAPH, content="In Exercises 1.23-1.25, find the eigenvalues.", id=0, seg_index=0, role="instruction"),
        ASTNode(type=NodeType.LIST, content="1.23 A", id=1, seg_index=0),
        ASTNode(type=NodeType.LIST, content="1.24 B", id=2, seg_index=0),
        ASTNode(type=NodeType.LIST, content="1.25 C", id=3, seg_index=0),
        ASTNode(type=NodeType.LIST, content="1.26 D", id=4, seg_index=0),
    ]


def _problems():
    return [
        Entity(type=EntityType.PROBLEM, members=[1], number="1.23"),
        Entity(type=EntityType.PROBLEM, members=[2], number="1.24"),
        Entity(type=EntityType.PROBLEM, members=[3], number="1.25"),
        Entity(type=EntityType.PROBLEM, members=[4], number="1.26"),
    ]


def test_stamps_only_the_in_range_problems():
    ranger = _ScriptedRanger({"1.23-1.25": ("1.23", "1.25", "find the eigenvalues")})
    problems = _problems()
    asyncio.run(distribute_instructions(_nodes(), problems, module=ranger))
    assert [p.instruction for p in problems] == [
        "find the eigenvalues", "find the eigenvalues", "find the eigenvalues", None,
    ]


def test_no_range_stamps_nothing():
    # Range-only MVP: a lead-in the ranger returns no range for governs nobody.
    ranger = _ScriptedRanger({"1.23-1.25": ("", "", "")})
    problems = _problems()
    asyncio.run(distribute_instructions(_nodes(), problems, module=ranger))
    assert all(p.instruction is None for p in problems)


def test_only_problems_after_the_lead_in_are_governed():
    # A same-numbered problem BEFORE the lead-in must not be stamped.
    nodes = [
        ASTNode(type=NodeType.LIST, content="1.24 earlier", id=0, seg_index=0),
        ASTNode(type=NodeType.PARAGRAPH, content="In Exercises 1.23-1.25, do X.", id=1, seg_index=0, role="instruction"),
        ASTNode(type=NodeType.LIST, content="1.24 later", id=2, seg_index=0),
    ]
    problems = [
        Entity(type=EntityType.PROBLEM, members=[0], number="1.24"),  # before the lead-in
        Entity(type=EntityType.PROBLEM, members=[2], number="1.24"),  # after the lead-in
    ]
    ranger = _ScriptedRanger({"1.23-1.25": ("1.23", "1.25", "do X")})
    asyncio.run(distribute_instructions(nodes, problems, module=ranger))
    assert [p.instruction for p in problems] == [None, "do X"]


def test_dotted_numbers_compare_component_wise():
    # 1.3 is NOT in 1.23-1.25 (float 1.3 would wrongly land between them).
    assert _parse_number("1.3") < _parse_number("1.23")
    nodes = [
        ASTNode(type=NodeType.PARAGRAPH, content="In Exercises 1.23-1.25, do X.", id=0, seg_index=0, role="instruction"),
        ASTNode(type=NodeType.LIST, content="1.3 outside", id=1, seg_index=0),
    ]
    problems = [Entity(type=EntityType.PROBLEM, members=[1], number="1.3")]
    ranger = _ScriptedRanger({"1.23-1.25": ("1.23", "1.25", "do X")})
    asyncio.run(distribute_instructions(nodes, problems, module=ranger))
    assert problems[0].instruction is None


def test_node_writes_problem_entities_channel_and_is_a_noop_without_lead_ins():
    # No tagged lead-in -> nothing to do, channel returned unchanged.
    nodes = [ASTNode(type=NodeType.LIST, content="1.23 A", id=0, seg_index=0)]
    problems = [Entity(type=EntityType.PROBLEM, members=[0], number="1.23")]
    node = InstructionDistributorNode(module=_ScriptedRanger({}))
    out = asyncio.run(node.run({"nodes": nodes, "problem_entities": problems}))
    assert set(out) == {"problem_entities"}
    assert out["problem_entities"][0].instruction is None
