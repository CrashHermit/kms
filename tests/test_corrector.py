"""Correction pass — pure dispatch/collect + the divergence guard. No network/LLM."""

from module.state import Segment
from module.corrector import CorrectorNode, _within_tolerance

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
