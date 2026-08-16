import asyncio

import dspy

from kms.construction import corrector
from kms.core import models

SENTINEL = object()


def _segment(index, content, image_path='/pages/Segment.png'):
    return models.Segment(index=index, image_path=image_path, content=content)


def test_worker_takes_the_correction_exactly_as_returned():
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
    segments = [
        _segment(0, 'definition with $x^2$'),
        _segment(1, 'plain prose, no math at all'),
        _segment(2, None),
        _segment(3, 'content but', image_path=''),
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


def test_specialist_prompts_describe_narrow_edit_contracts():
    for signature in (
        corrector.MathSignature,
        corrector.ProseSignature,
        corrector.LayoutSignature,
    ):
        prompt = signature.__doc__
        assert 'numbered' in prompt
        assert 'line edits' in prompt
        assert 'empty list' in prompt


def _dummy_lm() -> dspy.LM:
    return dspy.LM('openai/dummy', api_key='x')


def test_aforward_applies_consolidated_specialist_edits():
    class _FakeSpecialist:
        def __init__(self, edits):
            self.edits = edits

        async def aforward(self, **kwargs):
            assert kwargs['page_image'] is SENTINEL
            assert kwargs['transcription'] == 'a\nx_2\nc'
            return self.edits

    module = corrector.Corrector(language_model=_dummy_lm())
    module.math = _FakeSpecialist(
        [corrector.LineEdit(index=2, replacement='x^2')]
    )
    module.prose = _FakeSpecialist([])
    module.layout = _FakeSpecialist([])
    out = asyncio.run(
        module.aforward(page_image=SENTINEL, transcription='a\nx_2\nc')
    )
    assert out == 'a\nx^2\nc'


def test_aforward_returns_transcription_unchanged_when_no_edits():
    class _FakeSpecialist:
        async def aforward(self, **kwargs):
            return []

    module = corrector.Corrector(language_model=_dummy_lm())
    module.math = _FakeSpecialist()
    module.prose = _FakeSpecialist()
    module.layout = _FakeSpecialist()
    out = asyncio.run(
        module.aforward(page_image=SENTINEL, transcription='a\nb\nc')
    )
    assert out == 'a\nb\nc'


def test_conflicting_proposals_keep_the_first_and_log(caplog):
    edits = [
        corrector.LineEdit(index=1, replacement='first'),
        corrector.LineEdit(index=1, replacement='second'),
    ]
    with caplog.at_level('WARNING'):
        result = corrector.consolidate_edits(edits, line_count=1)
    assert result == [corrector.LineEdit(index=1, replacement='first')]
    assert 'conflicting correction proposals' in caplog.text
