"""Entity grouping: windowing, context, reconciliation, span resolution, and collect."""

import asyncio

from module.state import ASTNode, NodeType, EntityType, Entity, Member
from module.entity_grouper import (
    EntityGrouperNode,
    _windows,
    _context_before,
    _context_after,
    _reconcile,
)


def _para(i, text="x" * 40):
    return ASTNode(type=NodeType.PARAGRAPH, content=text, id=i, seg_index=0)


def test_windows_partition_whole_nodes_with_soft_budget():
    nodes = [_para(i) for i in range(5)] + [_para(5, "y" * 4000)]  # last is oversized
    w = _windows(nodes, budget=30)
    assert all(end > start for start, end in w)
    assert w[-1] == (5, 6)  # oversized node forms its own window, never split
    assert [i for s, e in w for i in range(s, e)] == list(range(6))  # exact partition


def test_context_helpers_bounded_and_empty_at_stream_ends():
    nodes = [_para(i) for i in range(5)]
    assert _context_before(nodes, 0, 500) is None
    assert _context_after(nodes, 5, 500) is None
    assert _context_before(nodes, 3, 500) is not None


def test_reconcile_chains_three_window_theorem_and_opener_type_wins():
    ents = [
        Entity(type=EntityType.THEOREM, members=[Member(10)], tail_open=True),
        Entity(type=EntityType.THEOREM, members=[Member(11)], head_continuation=True, tail_open=True),
        Entity(type=EntityType.THEOREM, members=[Member(12)], head_continuation=True),
        Entity(type=EntityType.DEFINITION, members=[Member(13)]),
    ]
    out = _reconcile(ents)
    assert [(e.type.value, [m.node_id for m in e.members]) for e in out] == [
        ("theorem", [10, 11, 12]),
        ("definition", [13]),
    ]


def test_reconcile_does_not_merge_without_tail_open():
    a = Entity(type=EntityType.DEFINITION, members=[Member(1)])
    b = Entity(type=EntityType.THEOREM, members=[Member(2)], head_continuation=True)
    assert len(_reconcile([a, b])) == 2


class _Span:
    def __init__(self, t, a, b, cb=False, ca=False):
        self.type, self.start, self.end = t, a, b
        self.continues_before, self.continues_after = cb, ca


class _Mod:
    def __init__(self, spans):
        self._spans = spans

    async def aforward(self, **_):
        class _P:
            entities = self._spans
        return _P()


def test_worker_gathered_problem_span_excludes_atomic_problem_node():
    current = [
        ASTNode(type=NodeType.PARAGRAPH, content="Example 1.", id=10, seg_index=0),
        ASTNode(type=NodeType.PROBLEM, content="99. atomic", id=11, seg_index=0),
        ASTNode(type=NodeType.PARAGRAPH, content="Solution", id=12, seg_index=0),
    ]
    node = EntityGrouperNode(module=_Mod([_Span("problem", 0, 2)]))
    out = asyncio.run(node.worker(
        {"window_index": 0, "current": current, "prev_context": None, "next_context": None}
    ))
    entities = out["entity_results"][0][1]
    assert len(entities) == 1
    assert entities[0].type == EntityType.PROBLEM
    assert [m.node_id for m in entities[0].members] == [10, 12]  # atomic problem node dropped


def test_collect_folds_gathered_and_wrapped_problems_in_document_order():
    nodes = [
        ASTNode(type=NodeType.PARAGRAPH, content="Thm", id=1, seg_index=0),
        ASTNode(type=NodeType.PARAGRAPH, content="pf", id=2, seg_index=0),
        ASTNode(type=NodeType.PROBLEM, content="1. do", id=3, seg_index=0),
        ASTNode(type=NodeType.PARAGRAPH, content="Def", id=4, seg_index=0),
    ]
    node = EntityGrouperNode(module=object())
    results = [
        (0, [Entity(type=EntityType.THEOREM, members=[Member(1), Member(2)])]),
        (1, [Entity(type=EntityType.DEFINITION, members=[Member(4)])]),
    ]
    out = node.collect({"nodes": nodes, "entity_results": results})["entities"]
    assert [(e.id, e.type.value, [m.node_id for m in e.members]) for e in out] == [
        (0, "theorem", [1, 2]),
        (1, "problem", [3]),  # atomic problem wrapped 1:1, slotted by position
        (2, "definition", [4]),
    ]
