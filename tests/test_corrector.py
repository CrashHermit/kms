"""Correction pass — pure dispatch/collect + the divergence guard. No network/LLM."""

from module.state import Segment
from module.corrector import CorrectorNode, _within_tolerance, _normalize_math_delimiters

# Keeps the node's pure dispatch/collect off the real (vision) LLM constructor.
SENTINEL = object()


def _seg(index, content, image_path="/pages/Segment.png"):
    return Segment(index=index, image_path=image_path, content=content)


def test_within_tolerance_accepts_light_edits_rejects_runaways():
    orig = "x" * 100
    assert _within_tolerance(orig, "x" * 100)      # identical
    assert _within_tolerance(orig, "x" * 80)       # −20%, a real fix
    assert _within_tolerance(orig, "x" * 130)      # +30% boundary
    assert not _within_tolerance(orig, "x" * 50)   # truncation
    assert not _within_tolerance(orig, "x" * 200)  # runaway rewrite
    assert not _within_tolerance(orig, "")         # empty
    assert not _within_tolerance(orig, "   ")      # whitespace only


def test_normalize_math_delimiters_swaps_display_and_inline():
    # \[ … \] -> $$ … $$ (display), \( … \) -> $ … $ (inline), whitespace preserved.
    assert _normalize_math_delimiters(r"\[ a^2 + b^2 \]") == "$$ a^2 + b^2 $$"
    assert _normalize_math_delimiters(r"see \(x_1\) here") == "see $x_1$ here"
    # multi-line display block (e.g. a wrapped array) keeps its interior verbatim.
    src = "\\[\n\\begin{array}{l} x \\end{array}\n\\]"
    assert _normalize_math_delimiters(src) == "$$\n\\begin{array}{l} x \\end{array}\n$$"


def test_normalize_math_delimiters_leaves_dollars_and_prose_untouched():
    # Already-correct `$$`/`$` and plain prose (incl. plain brackets/parens) are unchanged.
    already = "inline $x$ and display $$y$$ with a list item [a] and (b)"
    assert _normalize_math_delimiters(already) == already


def test_worker_output_is_delimiter_normalized_when_correction_rejected():
    # A runaway correction is rejected (kept original), but the kept text is still normalized.
    # image_path="" -> load_dspy_image returns None, so the worker needs no image file on disk.
    seg = _seg(0, r"kept \(x\) original", image_path="")

    class _RunawayModule:
        async def aforward(self, page_image, transcription):
            import dspy
            return dspy.Prediction(corrected="x" * 10_000)  # rejected by the guard

    import asyncio
    out = asyncio.run(CorrectorNode(module=_RunawayModule()).worker({"segment": seg}))
    assert out["correction_results"] == [(0, "kept $x$ original")]


def test_dispatch_proofreads_every_page_with_content_and_image():
    # No math gate: a prose page (seg 1) is proofread just like a math page (seg 0).
    segs = [
        _seg(0, "definition with $x^2$"),
        _seg(1, "plain prose, no math at all"),
        _seg(2, None),                       # no content -> skip
        _seg(3, "content but", image_path=""),  # no page image to check against -> skip
    ]
    sends = CorrectorNode(module=SENTINEL).dispatch({"segments": segs})
    assert sorted(s.arg["segment"].index for s in sends) == [0, 1]


def test_dispatch_falls_back_to_collect_when_none_eligible():
    node = CorrectorNode(module=SENTINEL)
    segs = [_seg(0, None), _seg(1, "x", image_path="")]
    assert node.dispatch({"segments": segs}) == "corrector_collect"


def test_collect_writes_corrected_back_and_leaves_others_untouched():
    segs = [_seg(0, "orig0"), _seg(1, "orig1")]
    out = CorrectorNode(module=SENTINEL).collect(
        {"segments": segs, "correction_results": [(0, "fixed0")]}
    )
    assert out["segments"][0].content == "fixed0"
    assert out["segments"][1].content == "orig1"
