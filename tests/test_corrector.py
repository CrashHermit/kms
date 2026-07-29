"""Correction pass — pure dispatch/collect + the divergence guard. No network/LLM."""

from kms.core import models
from kms.ingestion import corrector

# Keeps the node's pure dispatch/collect off the real (vision) LLM constructor.
SENTINEL = object()


def _seg(index, content, image_path='/pages/models.Segment.png'):
    return models.Segment(index=index, image_path=image_path, content=content)


def test_within_tolerance_accepts_light_edits_rejects_runaways():
    orig = 'x' * 100
    assert corrector._within_tolerance(orig, 'x' * 100)  # identical
    assert corrector._within_tolerance(orig, 'x' * 80)  # −20%, a real fix
    assert corrector._within_tolerance(orig, 'x' * 130)  # +30% boundary
    assert not corrector._within_tolerance(orig, 'x' * 50)  # truncation
    assert not corrector._within_tolerance(orig, 'x' * 200)  # runaway rewrite
    assert not corrector._within_tolerance(orig, '')  # empty
    assert not corrector._within_tolerance(orig, '   ')  # whitespace only


def test_normalize_math_delimiters_swaps_display_and_inline():
    # \[ … \] -> $$…$$ (display), \( … \) -> $…$ (inline). The spaces Mistral writes just
    # inside the escape delimiters go away, so an expression has one spelling.
    assert (
        corrector._normalize_math_delimiters(r'\[ a^2 + b^2 \]')
        == '$$a^2 + b^2$$'
    )
    assert (
        corrector._normalize_math_delimiters(r'see \(x_1\) here')
        == 'see $x_1$ here'
    )
    assert (
        corrector._normalize_math_delimiters(r'a \( 2^{p}-1 \) b')
        == 'a $2^{p}-1$ b'
    )
    # multi-line display block (e.g. a wrapped array) keeps its newlines — only the
    # horizontal padding is removed.
    src = '\\[\n\\begin{array}{l} x \\end{array}\n\\]'
    assert (
        corrector._normalize_math_delimiters(src)
        == '$$\n\\begin{array}{l} x \\end{array}\n$$'
    )


def test_normalize_math_delimiters_ignores_currency_prose():
    # A page with no escape-form delimiters is returned untouched, so two currency
    # amounts are never paired up as if the `$`s delimited math.
    prose = 'it cost $5 and then $6 later'
    assert corrector._normalize_math_delimiters(prose) == prose


def test_normalize_math_delimiters_leaves_dollars_and_prose_untouched():
    # Already-correct `$$`/`$` and plain prose (incl. plain brackets/parens) are unchanged.
    already = 'inline $x$ and display $$y$$ with a list item [a] and (b)'
    assert corrector._normalize_math_delimiters(already) == already


def test_worker_output_is_delimiter_normalized_when_correction_rejected():
    # A runaway correction is rejected (kept original), but the kept text is still normalized.
    # image_path="" -> _load_dspy_image returns None, so the worker needs no image file on disk.
    segment = _seg(0, r'kept \(x\) original', image_path='')

    class _RunawayModule:
        async def aforward(self, page_image, transcription):
            return 'x' * 10_000  # rejected by the guard

    import asyncio

    out = asyncio.run(
        corrector.CorrectorNode(module=_RunawayModule()).worker(
            {'segment': segment}
        )
    )
    assert out['correction_results'] == [(0, 'kept $x$ original')]


def test_dispatch_proofreads_every_page_with_content_and_image():
    # No math gate: a prose page (seg 1) is proofread just like a math page (seg 0).
    segs = [
        _seg(0, 'definition with $x^2$'),
        _seg(1, 'plain prose, no math at all'),
        _seg(2, None),  # no content -> skip
        _seg(
            3, 'content but', image_path=''
        ),  # no page image to check against -> skip
    ]
    sends = corrector.CorrectorNode(module=SENTINEL).dispatch(
        {'segments': segs}
    )
    assert sorted(s.arg['segment'].index for s in sends) == [0, 1]


def test_dispatch_falls_back_to_collect_when_none_eligible():
    node = corrector.CorrectorNode(module=SENTINEL)
    segs = [_seg(0, None), _seg(1, 'x', image_path='')]
    assert node.dispatch({'segments': segs}) == 'corrector_collect'


def test_collect_writes_corrected_back_and_leaves_others_untouched():
    segs = [_seg(0, 'orig0'), _seg(1, 'orig1')]
    out = corrector.CorrectorNode(module=SENTINEL).collect(
        {'segments': segs, 'correction_results': [(0, 'fixed0')]}
    )
    assert out['segments'][0].content == 'fixed0'
    assert out['segments'][1].content == 'orig1'
