"""Stage logging: the shared formatting helpers, and the per-stage INFO
summaries.

The pipeline previously emitted a single line for a whole run, which made a live
validation sweep reconstruct stage behaviour from the LangGraph State and the
DSPy cache (see ``robustness_test/ENTITY-REBUILD-VALIDATION.md``). These tests
pin the summaries that replaced that, in particular the two counts that make a
bad run diagnosable: the pedagogical component finder's procedure-span count and
the statement extractor's induced-type histogram.
"""

import asyncio
import logging

from kms.core import logs, models
from kms.ingestion import (
    pedagogical_component_finder,
    role_typer,
    seam_merger,
)

# --- logs.elide ---


def test_elide_collapses_whitespace_and_newlines():
    assert logs.elide('a\n  b\tc') == 'a b c'


def test_elide_truncates_with_an_ellipsis():
    assert logs.elide('abcdef', limit=3) == 'abc…'


def test_elide_keeps_text_at_the_limit_unmarked():
    assert logs.elide('abc', limit=3) == 'abc'


def test_elide_renders_none_as_empty():
    assert logs.elide(None) == ''


# --- logs.counts ---


def test_counts_orders_by_frequency_then_name():
    assert logs.counts(['b', 'a', 'b']) == 'b=2 a=1'


def test_counts_renders_none_values_and_empty_input():
    assert logs.counts([None]) == '?=1'
    assert logs.counts([]) == 'none'


# --- stage summaries ---


def _nodes(*contents):
    return [
        models.ParagraphNode(content=text, id=i, segment_index=0)
        for i, text in enumerate(contents)
    ]


class _ScriptedFinder:
    def __init__(self, scripted):
        self._scripted = list(scripted)

    async def aforward(self, current_nodes):
        return self._scripted.pop(0) if self._scripted else []


def test_pedagogical_component_finder_logs_the_span_count(caplog):
    module = _ScriptedFinder(
        [
            [
                pedagogical_component_finder.Span(start=0, end=0),
                pedagogical_component_finder.Span(start=1, end=1),
            ],
            [],
        ]
    )
    with caplog.at_level(
        logging.INFO, logger='kms.ingestion.pedagogical_component_finder'
    ):
        asyncio.run(
            pedagogical_component_finder.find_spans(
                _nodes('Theorem 1', 'Proof.', 'tail'), module=module
            )
        )
    assert '3 nodes -> 2 span(s)' in caplog.text


class _ScriptedRoles:
    def __init__(self, roles):
        self._roles = list(roles)

    async def acall(self, contents):
        return [self._roles.pop(0)]


def test_role_typer_logs_the_block_derivation_split(caplog):
    # "statements found but zero derivations" is the signature of an
    # unmarked-derivation miss,
    # so the two counts must be reported separately.
    node = role_typer.RoleTyperNode(
        module=_ScriptedRoles(['statement', 'procedure'])
    )
    with caplog.at_level(logging.INFO, logger='kms.ingestion.role_typer'):
        out = asyncio.run(
            node.run({'nodes': _nodes('a', 'b'), 'spans': [[0], [1]]})
        )
    assert '2 span(s) -> 1 statement(s), 1 procedure(s)' in caplog.text
    assert [p.block for p in out['procedures']] == [[1]]


def test_role_typer_logs_zero_derivations(caplog):
    node = role_typer.RoleTyperNode(module=_ScriptedRoles(['statement']))
    with caplog.at_level(logging.INFO, logger='kms.ingestion.role_typer'):
        asyncio.run(node.run({'nodes': _nodes('a'), 'spans': [[0]]}))
    assert '1 statement(s), 0 procedure(s)' in caplog.text


def test_seam_merger_logs_the_flattened_stream_size(caplog):
    segment = models.Segment(index=0, image_path='p0.png')
    segment.nodes = _nodes('a', 'b')
    node = seam_merger.SeamMergerNode()
    with caplog.at_level(logging.INFO, logger='kms.ingestion.seam_merger'):
        result = node.odd_collect(
            {'segments': [segment], 'seam_odd_results': []}
        )
    assert len(result['nodes']) == 2
    assert '1 page(s) -> flat stream of 2 node(s)' in caplog.text
