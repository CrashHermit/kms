"""Problem finder: the core cursor-walk banking rule and the graph-node wrapper."""

import asyncio

from kms.core.state import ASTNode, EntityType, NodeType
from kms.entity.finders.problem import ProblemFinderNode, ProblemSpan, find_problems


def _nodes():
    return [
        ASTNode(type=NodeType.PARAGRAPH, content="intro prose", id=0, seg_index=0),
        ASTNode(type=NodeType.HEADER, content="Example 1", id=1, seg_index=0),
        ASTNode(type=NodeType.PARAGRAPH, content="solve this", id=2, seg_index=0),
        ASTNode(type=NodeType.PARAGRAPH, content="more prose", id=3, seg_index=0),
    ]


class _ScriptedFinder:
    """A stand-in Module whose aforward returns pre-scripted spans per call."""

    def __init__(self, scripted):
        self._scripted = list(scripted)

    async def aforward(self, current_nodes):
        return self._scripted.pop(0) if self._scripted else []


def test_find_problems_banks_a_bounded_problem_and_emits_member_ids():
    # First read spans the Example (local positions 1-2); node 3 follows it, so it is
    # bounded and banked. The cursor then advances past it and the tail read is empty.
    module = _ScriptedFinder([[ProblemSpan(start=1, end=2)], []])
    problems = asyncio.run(find_problems(_nodes(), module=module))
    assert len(problems) == 1
    assert problems[0].type == EntityType.PROBLEM
    assert problems[0].members == [1, 2]  # stable global ids, not window positions


def test_find_problems_on_prose_only_stream_returns_nothing():
    module = _ScriptedFinder([[]])
    assert asyncio.run(find_problems(_nodes(), module=module)) == []


def test_node_run_writes_the_problem_channel():
    node = ProblemFinderNode(module=_ScriptedFinder([[ProblemSpan(start=1, end=2)], []]))
    out = asyncio.run(node.run({"nodes": _nodes()}))
    assert list(out.keys()) == ["problem_entities"]
    assert [e.members for e in out["problem_entities"]] == [[1, 2]]


def test_node_run_on_empty_stream_yields_empty_channel():
    node = ProblemFinderNode(module=_ScriptedFinder([]))
    assert asyncio.run(node.run({"nodes": []})) == {"problem_entities": []}
