"""Stage logging: the shared formatting helpers, and the per-stage INFO summaries.

The pipeline previously emitted a single line for a whole run, which made a live
validation sweep reconstruct stage behaviour from the LangGraph State and the DSPy cache
(see ``robustness_test/ENTITY-REBUILD-VALIDATION.md``). These tests pin the summaries that
replaced that, in particular the two counts that make a bad run diagnosable: the group
finder's procedure-span count and the statement extractor's induced-type histogram.
"""

import asyncio
import logging

from kms.core import logs, models
from kms.entity.group_finder import Span, find_groups
from kms.entity.instruction_finder import tag_instructions
from kms.entity.statement_extractor import StatementExtractorNode
from kms.ingestion.seam_merger import SeamMergerNode

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
        models.ASTNode(
            type=models.NodeType.PARAGRAPH, content=text, id=i, segment_index=0
        )
        for i, text in enumerate(contents)
    ]


class _ScriptedFinder:
    def __init__(self, scripted):
        self._scripted = list(scripted)

    async def aforward(self, current_nodes):
        return self._scripted.pop(0) if self._scripted else []


def test_group_finder_logs_block_and_procedure_span_counts(caplog):
    # A block plus its derivation: the summary must report both roles separately, since
    # "blocks found but zero procedure spans" is the signature of an unmarked-derivation
    # miss and is otherwise invisible.
    module = _ScriptedFinder(
        [
            [
                Span(start=0, end=0, role='entity'),
                Span(start=1, end=1, role='procedure'),
            ],
            [],
        ]
    )
    with caplog.at_level(logging.INFO, logger='kms.entity.group_finder'):
        asyncio.run(
            find_groups(_nodes('Theorem 1', 'Proof.', 'tail'), module=module)
        )
    assert '3 nodes -> 1 block(s), 1 procedure span(s)' in caplog.text


def test_group_finder_logs_zero_procedure_spans(caplog):
    module = _ScriptedFinder([[Span(start=0, end=0, role='entity')], []])
    with caplog.at_level(logging.INFO, logger='kms.entity.group_finder'):
        asyncio.run(find_groups(_nodes('Example 1', 'tail'), module=module))
    assert '1 block(s), 0 procedure span(s)' in caplog.text


class _ScriptedIdentity:
    def __init__(self, types):
        self._types = list(types)

    async def identity(self, members):
        from kms.entity.statement_extractor import Identity

        return Identity(type=self._types.pop(0), label=None, number=None)


def test_statement_extractor_logs_the_induced_type_histogram(caplog):
    entities = [models.Entity(members=[0]), models.Entity(members=[1])]
    node = StatementExtractorNode(
        module=_ScriptedIdentity(['theorem', 'theorem'])
    )
    with caplog.at_level(logging.INFO, logger='kms.entity.statement_extractor'):
        asyncio.run(node.run({'nodes': _nodes('a', 'b'), 'entities': entities}))
    assert '2 block(s) typed | theorem=2' in caplog.text


class _ScriptedTagger:
    def __init__(self, positions):
        self._positions = list(positions)

    async def aforward(self, current_nodes):
        return self._positions.pop(0) if self._positions else []


def test_instruction_finder_logs_the_tagged_lead_in_count(caplog):
    nodes = _nodes('For the following exercises, graph.', '1. graph it')
    with caplog.at_level(logging.INFO, logger='kms.entity.instruction_finder'):
        asyncio.run(tag_instructions(nodes, module=_ScriptedTagger([[0]])))
    assert '2 node(s) -> 1 lead-in(s) tagged' in caplog.text


def test_seam_merger_logs_the_flattened_stream_size(caplog):
    segment = models.Segment(index=0, image_path='p0.png')
    segment.nodes = _nodes('a', 'b')
    node = SeamMergerNode()
    with caplog.at_level(logging.INFO, logger='kms.ingestion.seam_merger'):
        result = node.odd_collect(
            {'segments': [segment], 'seam_odd_results': []}
        )
    assert len(result['nodes']) == 2
    assert '1 page(s) -> flat stream of 2 node(s)' in caplog.text
