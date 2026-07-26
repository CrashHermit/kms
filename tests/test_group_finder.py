"""Group finder: the core cursor-walk banking rule, the statement/derivation cut, the
one-partition guarantee, and the graph-node wrapper.

The finder emits UNTYPED spans — whether a span is a block or a derivation is `role_typer`'s
question, so nothing here asserts a role."""

import asyncio

from kms.core import models
from kms.entity.group_finder import (
    GroupFinderNode,
    Span,
    _clean_spans,
    find_spans,
)


def _nodes():
    return [
        models.ASTNode(
            type=models.NodeType.PARAGRAPH,
            content='intro prose',
            id=0,
            segment_index=0,
        ),
        models.ASTNode(
            type=models.NodeType.HEADER,
            content='Example 1',
            id=1,
            segment_index=0,
        ),
        models.ASTNode(
            type=models.NodeType.PARAGRAPH,
            content='solve this',
            id=2,
            segment_index=0,
        ),
        models.ASTNode(
            type=models.NodeType.PARAGRAPH,
            content='more prose',
            id=3,
            segment_index=0,
        ),
    ]


class _ScriptedFinder:
    """A stand-in Module whose aforward returns pre-scripted spans per call."""

    def __init__(self, scripted):
        self._scripted = list(scripted)

    async def aforward(self, current_nodes):
        return self._scripted.pop(0) if self._scripted else []


def test_banks_a_bounded_span_and_emits_member_ids():
    # First read spans the Example (local positions 1-2); node 3 follows it, so it is
    # bounded and banked. The cursor then advances past it and the tail read is empty.
    module = _ScriptedFinder([[Span(start=1, end=2)], []])
    spans = asyncio.run(find_spans(_nodes(), module=module))
    assert spans == [[1, 2]]  # stable global ids, not positions


def test_statement_and_derivation_are_cut_into_separate_spans():
    # A statement at 1 and its derivation at 2 — two adjacent spans, never fused. The cut
    # is made here (a boundary); which one is the derivation is decided downstream.
    module = _ScriptedFinder([[Span(start=1, end=1), Span(start=2, end=2)], []])
    assert asyncio.run(find_spans(_nodes(), module=module)) == [[1], [2]]


def test_on_prose_only_stream_returns_nothing():
    module = _ScriptedFinder([[]])
    assert asyncio.run(find_spans(_nodes(), module=module)) == []


# --- span cleaning: clamping and one partition ---


def test_clean_spans_clamps_into_the_window():
    cleaned = _clean_spans([Span(start=-5, end=99)], 2)
    assert (cleaned[0].start, cleaned[0].end) == (0, 2)


def test_clean_spans_drops_overlaps_so_a_node_belongs_to_one_span():
    cleaned = _clean_spans(
        [
            Span(start=0, end=2),
            Span(start=1, end=3),  # overlaps — dropped
            Span(start=3, end=3),
        ],
        3,
    )
    assert [(span.start, span.end) for span in cleaned] == [(0, 2), (3, 3)]


# --- graph node ---


def test_node_run_writes_the_spans_channel():
    node = GroupFinderNode(
        module=_ScriptedFinder(
            [[Span(start=1, end=1), Span(start=2, end=2)], []]
        )
    )
    out = asyncio.run(node.run({'nodes': _nodes()}))
    assert set(out) == {'spans'}
    assert out['spans'] == [[1], [2]]


def test_node_run_on_empty_stream_yields_an_empty_channel():
    node = GroupFinderNode(module=_ScriptedFinder([]))
    assert asyncio.run(node.run({'nodes': []})) == {'spans': []}
