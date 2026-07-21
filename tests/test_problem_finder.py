"""Problem finder: the LangGraph node wrapper and the core cursor-walk banking rule."""

import asyncio

from module.state import ASTNode, NodeType, EntityType
from module.problem_finder import ProblemFinderNode, ProblemSpan, find_problems

# A sentinel keeps the node's pure dispatch/collect off the real LLM constructor.
SENTINEL = object()


def _nodes():
    return [
        ASTNode(type=NodeType.PARAGRAPH, content="intro prose", id=0, seg_index=0),
        ASTNode(type=NodeType.HEADER, content="Example 1", id=1, seg_index=0),
        ASTNode(type=NodeType.PARAGRAPH, content="solve this", id=2, seg_index=0),
        ASTNode(type=NodeType.PARAGRAPH, content="more prose", id=3, seg_index=0),
    ]


def test_dispatch_fans_out_one_send_with_the_whole_stream():
    finder = ProblemFinderNode(module=SENTINEL)
    nodes = _nodes()
    sends = finder.dispatch({"nodes": nodes})
    assert len(sends) == 1  # the walk is one sequential unit, not sharded
    assert sends[0].arg["nodes"] is nodes


def test_dispatch_short_circuits_on_empty_stream():
    finder = ProblemFinderNode(module=SENTINEL)
    assert finder.dispatch({"nodes": []}) == "problem_finder_collect"


def test_collect_seeds_entities_from_finder_results():
    finder = ProblemFinderNode(module=SENTINEL)
    from module.state import Entity, Member
    problems = [Entity(type=EntityType.PROBLEM, members=[Member(1)])]
    out = finder.collect({"finder_results": [problems]})
    assert out["entities"] is problems
    assert finder.collect({})["entities"] == []  # no results -> empty overlay


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
    assert [m.node_id for m in problems[0].members] == [1, 2]  # stable global ids, not positions
