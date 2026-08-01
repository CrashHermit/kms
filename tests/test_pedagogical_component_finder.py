"""Pedagogical component finder: cursor-walk banking and span cutting."""

import asyncio

from kms.core import models
from kms.ingestion import pedagogical_component_finder


class _ScriptedFinder:
    def __init__(self, scripted):
        self._scripted = list(scripted)

    async def aforward(self, current_nodes):
        return self._scripted.pop(0) if self._scripted else []


def _nodes():
    return [
        models.ParagraphNode(content='intro prose', id=0),
        models.HeaderNode(content='Example 1', id=1),
        models.ParagraphNode(content='solve this', id=2),
        models.ParagraphNode(content='more prose', id=3),
    ]


def test_banks_a_bounded_span_and_emits_member_ids():
    module = _ScriptedFinder(
        [[pedagogical_component_finder.Span(start=1, end=2)], []]
    )
    spans = asyncio.run(
        pedagogical_component_finder.find_spans(_nodes(), module=module)
    )
    assert spans == [[1, 2]]


def test_banks_multiple_bounded_spans_in_document_order():
    module = _ScriptedFinder(
        [
            [
                pedagogical_component_finder.Span(start=1, end=1),
                pedagogical_component_finder.Span(start=2, end=2),
            ],
            [],
        ]
    )
    assert asyncio.run(
        pedagogical_component_finder.find_spans(_nodes(), module=module)
    ) == [[1], [2]]


def test_on_prose_only_stream_returns_nothing():
    module = _ScriptedFinder([[]])
    assert (
        asyncio.run(
            pedagogical_component_finder.find_spans(_nodes(), module=module)
        )
        == []
    )


def test_normalize_spans_clamps_into_the_window():
    cleaned = pedagogical_component_finder._normalize_spans(
        [pedagogical_component_finder.Span(start=-5, end=99)], 2
    )
    assert (cleaned[0].start, cleaned[0].end) == (0, 2)


def test_normalize_spans_preserves_overlaps():
    cleaned = pedagogical_component_finder._normalize_spans(
        [
            pedagogical_component_finder.Span(start=0, end=2),
            pedagogical_component_finder.Span(start=1, end=3),
            pedagogical_component_finder.Span(start=3, end=3),
        ],
        3,
    )
    assert [(s.start, s.end) for s in cleaned] == [(0, 2), (1, 3), (3, 3)]


def test_node_run_writes_the_spans_channel():
    node = pedagogical_component_finder.PedagogicalComponentFinderNode(
        module=_ScriptedFinder(
            [
                [
                    pedagogical_component_finder.Span(start=1, end=1),
                    pedagogical_component_finder.Span(start=2, end=2),
                ],
                [],
            ]
        )
    )
    out = asyncio.run(node.run({'nodes': _nodes()}))
    assert set(out) == {'spans'}
    assert out['spans'] == [[1], [2]]


def test_node_run_on_empty_stream_yields_an_empty_channel():
    node = pedagogical_component_finder.PedagogicalComponentFinderNode(
        module=_ScriptedFinder([])
    )
    assert asyncio.run(node.run({'nodes': []})) == {'spans': []}
