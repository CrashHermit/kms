"""Group finder: the core cursor-walk banking rule, the entity/procedure span split, the
one-partition guarantee, and the graph-node wrapper."""

import asyncio

from kms.core import models
from kms.entity.group_finder import (
    GroupFinderNode,
    Span,
    _clean_spans,
    find_groups,
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


def test_banks_a_bounded_block_and_emits_member_ids():
    # First read spans the Example (local positions 1-2); node 3 follows it, so it is
    # bounded and banked. The cursor then advances past it and the tail read is empty.
    module = _ScriptedFinder([[Span(start=1, end=2, role='entity')], []])
    entities, procedures = asyncio.run(find_groups(_nodes(), module=module))
    assert len(entities) == 1
    assert entities[0].members == [1, 2]  # stable global ids, not positions
    assert entities[0].type is None  # the finder emits spans only, never a type
    assert procedures == []


def test_statement_and_procedure_are_separate_spans():
    # A block at 1 and its derivation at 2 — two adjacent spans, never fused.
    module = _ScriptedFinder(
        [
            [
                Span(start=1, end=1, role='entity'),
                Span(start=2, end=2, role='procedure'),
            ],
            [],
        ]
    )
    entities, procedures = asyncio.run(find_groups(_nodes(), module=module))
    assert [entity.members for entity in entities] == [[1]]
    assert procedures == [[2]]


def test_on_prose_only_stream_returns_nothing():
    module = _ScriptedFinder([[]])
    assert asyncio.run(find_groups(_nodes(), module=module)) == ([], [])


# --- span cleaning: clamping, role fallback, one partition ---


def test_clean_spans_clamps_into_the_window():
    cleaned = _clean_spans([Span(start=-5, end=99, role='entity')], 2)
    assert (cleaned[0].start, cleaned[0].end) == (0, 2)


def test_clean_spans_falls_back_to_entity_for_an_unknown_role():
    cleaned = _clean_spans([Span(start=0, end=0, role='nonsense')], 2)
    assert cleaned[0].role == 'entity'


def test_clean_spans_drops_overlaps_so_a_node_belongs_to_one_span():
    cleaned = _clean_spans(
        [
            Span(start=0, end=2, role='entity'),
            Span(start=1, end=3, role='entity'),  # overlaps — dropped
            Span(start=3, end=3, role='procedure'),
        ],
        3,
    )
    assert [(span.start, span.end) for span in cleaned] == [(0, 2), (3, 3)]


# --- graph node ---


def test_node_run_writes_both_channels():
    node = GroupFinderNode(
        module=_ScriptedFinder(
            [
                [
                    Span(start=1, end=1, role='entity'),
                    Span(start=2, end=2, role='procedure'),
                ],
                [],
            ]
        )
    )
    out = asyncio.run(node.run({'nodes': _nodes()}))
    assert set(out) == {'entities', 'procedure_spans'}
    assert [entity.members for entity in out['entities']] == [[1]]
    assert out['procedure_spans'] == [[2]]


def test_node_run_on_empty_stream_yields_empty_channels():
    node = GroupFinderNode(module=_ScriptedFinder([]))
    assert asyncio.run(node.run({'nodes': []})) == {
        'entities': [],
        'procedure_spans': [],
    }
