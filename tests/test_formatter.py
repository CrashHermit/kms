import asyncio

import pytest

from kms.construction import formatter
from kms.core import models

SENTINEL = object()


def _document(index, content):
    return models.Document(
        index=index,
        image_path='',
        content=content,
        nodes=[models.Node(index=0, content=content)] if content else [],
    )


def test_worker_takes_the_result_exactly_as_returned():
    document = _document(0, r'inline \(x\) here')

    class _Module:
        async def aforward(self, node_content):
            assert node_content == r'inline \(x\) here'
            return 'inline $x$ here'

    out = asyncio.run(
        formatter.FormatterNode(module=_Module()).worker(
            {'document': document, 'node': document.nodes[0]}
        )
    )
    assert out['format_results'] == [(0, 0, 'inline $x$ here')]


def test_worker_receives_the_node_content():
    document = _document(3, '# Heading')
    seen = []

    class _Module:
        async def aforward(self, node_content):
            seen.append(node_content)
            return node_content

    asyncio.run(
        formatter.FormatterNode(module=_Module()).worker(
            {'document': document, 'node': document.nodes[0]}
        )
    )
    assert seen == ['# Heading']


def test_dispatch_formats_every_node_with_content():
    documents = [
        _document(0, 'display $$y$$'),
        _document(1, 'plain prose'),
        _document(2, None),
        _document(3, ''),
    ]
    sends = formatter.FormatterNode(module=SENTINEL).dispatch(
        {'documents': documents}
    )
    assert sorted(
        (s.arg['document'].index, s.arg['node'].index) for s in sends
    ) == [(0, 0), (1, 0)]


def test_dispatch_falls_back_to_collect_when_none_eligible():
    node = formatter.FormatterNode(module=SENTINEL)
    documents = [_document(0, None), _document(1, '')]
    assert node.dispatch({'documents': documents}) == 'formatter_collect'


def test_dispatch_handles_missing_segments():
    node = formatter.FormatterNode(module=SENTINEL)
    assert node.dispatch({}) == 'formatter_collect'


def test_collect_writes_formatted_back_and_leaves_others_untouched():
    documents = [_document(0, 'orig0'), _document(1, 'orig1')]
    out = formatter.FormatterNode(module=SENTINEL).collect(
        {
            'documents': documents,
            'format_results': [(0, 0, 'formatted0')],
        }
    )
    assert out['documents'][0].nodes[0].content == 'formatted0'
    assert out['documents'][1].nodes[0].content == 'orig1'


def test_collect_is_a_noop_without_results():
    documents = [_document(0, 'orig0')]
    out = formatter.FormatterNode(module=SENTINEL).collect(
        {'documents': documents}
    )
    assert out['documents'][0].nodes[0].content == 'orig0'


def test_prompt_retains_node_content_rules():
    prompt = formatter.Signature.__doc__
    assert 'one canonical document node (block)' in prompt
    assert 'one page of a document' not in prompt
    assert '![N]()' in prompt
    for retained in ('Order.', 'Numbering and labels.', 'Code and verbatim'):
        assert retained in prompt
    assert 'Page furniture.' not in prompt
    assert 'halves of one equation are joined' not in prompt


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
