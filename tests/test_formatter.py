import asyncio

import pytest

from kms.construction import formatter
from kms.core import models

SENTINEL = object()


def _segment(index, content):
    return models.Segment(index=index, image_path='', content=content)


def test_worker_takes_the_result_exactly_as_returned():
    segment = _segment(0, r'inline \(x\) here')

    class _Module:
        async def aforward(self, markdown):
            return 'inline $x$ here'

    out = asyncio.run(
        formatter.FormatterNode(module=_Module()).worker({'segment': segment})
    )
    assert out['format_results'] == [(0, 'inline $x$ here')]


def test_worker_receives_the_page_markdown():
    segment = _segment(3, '# Heading')
    seen = []

    class _Module:
        async def aforward(self, markdown):
            seen.append(markdown)
            return markdown

    asyncio.run(
        formatter.FormatterNode(module=_Module()).worker({'segment': segment})
    )
    assert seen == ['# Heading']


def test_dispatch_formats_every_page_with_content():
    segments = [
        _segment(0, 'display $$y$$'),
        _segment(1, 'plain prose'),
        _segment(2, None),
        _segment(3, ''),
    ]
    sends = formatter.FormatterNode(module=SENTINEL).dispatch(
        {'segments': segments}
    )
    assert sorted(s.arg['segment'].index for s in sends) == [0, 1]


def test_dispatch_falls_back_to_collect_when_none_eligible():
    node = formatter.FormatterNode(module=SENTINEL)
    segments = [_segment(0, None), _segment(1, '')]
    assert node.dispatch({'segments': segments}) == 'formatter_collect'


def test_dispatch_handles_missing_segments():
    node = formatter.FormatterNode(module=SENTINEL)
    assert node.dispatch({}) == 'formatter_collect'


def test_collect_writes_formatted_back_and_leaves_others_untouched():
    segments = [_segment(0, 'orig0'), _segment(1, 'orig1')]
    out = formatter.FormatterNode(module=SENTINEL).collect(
        {'segments': segments, 'format_results': [(0, 'formatted0')]}
    )
    assert out['segments'][0].content == 'formatted0'
    assert out['segments'][1].content == 'orig1'


def test_collect_is_a_noop_without_results():
    segments = [_segment(0, 'orig0')]
    out = formatter.FormatterNode(module=SENTINEL).collect(
        {'segments': segments}
    )
    assert out['segments'][0].content == 'orig0'


def test_prompt_forbids_touching_figure_placeholders():
    prompt = formatter.Signature.__doc__
    assert '![N]()' in prompt
    for forbidden in ('Order.', 'Numbering and labels.', 'Code and verbatim'):
        assert forbidden in prompt


def test_prompt_joins_split_display_equations():
    prompt = formatter.Signature.__doc__
    assert 'halves of one equation are joined' in prompt
    assert 'relational operator' in prompt
    assert 'binary operator' in prompt
    assert 'back-to-back equations stay separate' in prompt


def test_number_lines_prefixes_each_line_with_its_index():
    assert formatter.number_lines('alpha\nbeta\n\ngamma') == (
        '[1] alpha\n[2] beta\n[3] \n[4] gamma'
    )


def test_apply_line_edits_replaces_a_single_line():
    edits = [formatter.LineEdit(index=2, replacement='B')]
    assert formatter.apply_line_edits('a\nb\nc', edits) == 'a\nB\nc'


def test_apply_line_edits_deletes_a_line_on_empty_replacement():
    edits = [formatter.LineEdit(index=2, replacement='')]
    assert formatter.apply_line_edits('a\nb\nc', edits) == 'a\nc'


def test_apply_line_edits_expands_a_line_on_multiline_replacement():
    edits = [formatter.LineEdit(index=2, replacement='B1\nB2')]
    assert formatter.apply_line_edits('a\nb\nc', edits) == 'a\nB1\nB2\nc'


def test_apply_line_edits_inserts_before_an_anchor():
    edits = [
        formatter.LineEdit(
            index=2, operation='insert_before', replacement='before'
        )
    ]
    assert formatter.apply_line_edits('a\nb\nc', edits) == 'a\nbefore\nb\nc'


def test_apply_line_edits_inserts_after_an_anchor():
    edits = [
        formatter.LineEdit(
            index=2, operation='insert_after', replacement='after'
        )
    ]
    assert formatter.apply_line_edits('a\nb\nc', edits) == 'a\nb\nafter\nc'


def test_apply_line_edits_deletes_with_explicit_operation():
    edits = [
        formatter.LineEdit(index=2, operation='delete', replacement='ignored')
    ]
    assert formatter.apply_line_edits('a\nb\nc', edits) == 'a\nc'


def test_apply_line_edits_applies_multiple_edits():
    edits = [
        formatter.LineEdit(index=1, replacement='A'),
        formatter.LineEdit(index=3, replacement='C'),
    ]
    assert formatter.apply_line_edits('a\nb\nc\nd', edits) == 'A\nb\nC\nd'


def test_apply_line_edits_rewrites_heading_and_deletes_underline():
    edits = [
        formatter.LineEdit(index=1, replacement='# Heading'),
        formatter.LineEdit(index=2, replacement=''),
    ]
    assert (
        formatter.apply_line_edits('Heading\n=======\nbody', edits)
        == '# Heading\nbody'
    )


def test_apply_line_edits_rejects_out_of_range_index():
    edits = [formatter.LineEdit(index=5, replacement='')]
    with pytest.raises(RuntimeError, match='out of range'):
        formatter.apply_line_edits('a\nb', edits)


def test_apply_line_edits_rejects_zero_index():
    edits = [formatter.LineEdit(index=0, replacement='')]
    with pytest.raises(RuntimeError, match='out of range'):
        formatter.apply_line_edits('a\nb', edits)


def test_apply_line_edits_rejects_duplicate_index():
    edits = [
        formatter.LineEdit(index=2, replacement=''),
        formatter.LineEdit(index=2, replacement=''),
    ]
    with pytest.raises(RuntimeError, match='duplicate'):
        formatter.apply_line_edits('a\nb\nc', edits)
