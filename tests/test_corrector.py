"""Correction pass — pure dispatch/collect. No network/LLM."""

import asyncio

from kms.core import models
from kms.ingestion import corrector

# Keeps the node's pure dispatch/collect off the real (vision) LLM constructor.
SENTINEL = object()


def _segment(index, content, image_path='/pages/Segment.png'):
    return models.Segment(index=index, image_path=image_path, content=content)


def test_worker_takes_the_correction_exactly_as_returned():
    # Nothing outside the model touches the page: the correction becomes the
    # page verbatim, however far it diverges and whatever its delimiters.
    # image_path="" -> _load_dspy_image returns None, so the worker needs no
    # image file on disk.
    segment = _segment(0, 'orig', image_path='')

    class _DivergentModule:
        async def aforward(self, page_image, transcription):
            return r'a much longer rewrite with \(x\) in it'

    out = asyncio.run(
        corrector.CorrectorNode(module=_DivergentModule()).worker(
            {'segment': segment}
        )
    )
    assert out['correction_results'] == [
        (0, r'a much longer rewrite with \(x\) in it')
    ]


def test_dispatch_proofreads_every_page_with_content_and_image():
    # No math gate: a prose page (page 1) is proofread just like a math page
    # (page 0).
    segments = [
        _segment(0, 'definition with $x^2$'),
        _segment(1, 'plain prose, no math at all'),
        _segment(2, None),  # no content -> skip
        _segment(
            3, 'content but', image_path=''
        ),  # no page image to check against -> skip
    ]
    sends = corrector.CorrectorNode(module=SENTINEL).dispatch(
        {'segments': segments}
    )
    assert sorted(s.arg['segment'].index for s in sends) == [0, 1]


def test_dispatch_falls_back_to_collect_when_none_eligible():
    node = corrector.CorrectorNode(module=SENTINEL)
    segments = [_segment(0, None), _segment(1, 'x', image_path='')]
    assert node.dispatch({'segments': segments}) == 'corrector_collect'


def test_collect_writes_corrected_back_and_leaves_others_untouched():
    segments = [_segment(0, 'orig0'), _segment(1, 'orig1')]
    out = corrector.CorrectorNode(module=SENTINEL).collect(
        {'segments': segments, 'correction_results': [(0, 'fixed0')]}
    )
    assert out['segments'][0].content == 'fixed0'
    assert out['segments'][1].content == 'orig1'
