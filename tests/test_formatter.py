"""Formatting pass — pure dispatch/collect. No network/LLM."""

import asyncio

from kms.core import models
from kms.ingestion import formatter

# Keeps the node's pure dispatch/collect off the real LLM constructor.
SENTINEL = object()


def _segment(index, content):
    # No page image: the formatter works from the markdown and the rules.
    return models.Segment(index=index, image_path='', content=content)


def test_worker_takes_the_result_exactly_as_returned():
    # Unguarded, like the corrector: nothing outside the model touches the page.
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
    # Unlike the corrector there is no image requirement, so a segment with no
    # page render still qualifies.
    segments = [
        _segment(0, 'display $$y$$'),
        _segment(1, 'plain prose'),
        _segment(2, None),  # no content -> skip
        _segment(3, ''),  # empty content -> skip
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
    # `![N]()` is resolved positionally against its page's extracted figures;
    # a formatter that rewrote one would silently lose that figure, so the
    # prompt must say so.
    prompt = formatter.Signature.__doc__
    assert '![N]()' in prompt
    for forbidden in ('Order.', 'Numbering and labels.', 'Code and verbatim'):
        assert forbidden in prompt


def test_prompt_joins_split_display_equations():
    # A display equation broken across two `$$` blocks by the OCR must be
    # rejoined: the equation extractor runs per-node downstream and expects
    # each MathNode to carry one complete equation.
    prompt = formatter.Signature.__doc__
    assert 'halves of one equation are joined' in prompt
    assert 'relational operator' in prompt
    assert 'binary operator' in prompt
    assert 'back-to-back equations stay separate' in prompt
