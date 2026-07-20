"""Mistral OCR response → Segment mapping. No network, no keys, no LLM — just the
pure transform that turns an OCR JSON response into the pipeline's Segment backbone."""

from pathlib import Path

from module.mistral_ocr import build_segments, _rewrite_page

# A 1x1 PNG, base64 — stands in for a returned figure crop.
_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR4"
    "2mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def _img(image_id: str, data_url: bool = False) -> dict:
    b64 = f"data:image/png;base64,{_PNG_B64}" if data_url else _PNG_B64
    return {
        "id": image_id,
        "image_base64": b64,
        "top_left_x": 0, "top_left_y": 0,
        "bottom_right_x": 1, "bottom_right_y": 1,
    }


def test_build_segments_rewrites_refs_and_saves_pictures(tmp_path):
    resp = {"pages": [{
        "index": 0,
        "markdown": "# Title\n\n![alt](img-0.jpeg)\n\nprose $x^2$\n\n![alt2](img-1.jpeg)\n",
        "images": [_img("img-0.jpeg"), _img("img-1.jpeg", data_url=True)],
    }]}
    segs = build_segments(resp, tmp_path)
    assert len(segs) == 1
    seg = segs[0]
    assert seg.index == 0
    # Mistral ids rewritten to the positional ![N]() convention, in reading order.
    assert "![1]()" in seg.content and "![2]()" in seg.content
    assert "img-0.jpeg" not in seg.content and "img-1.jpeg" not in seg.content
    assert [p.index for p in seg.pictures] == [1, 2]
    for p in seg.pictures:
        assert Path(p.image_path).exists() and Path(p.image_path).stat().st_size > 0


def test_unreferenced_figure_is_still_saved(tmp_path):
    # A figure that never appears inline in the markdown must not be silently dropped.
    resp = {"pages": [{"index": 0, "markdown": "prose only, no refs", "images": [_img("img-0.jpeg")]}]}
    segs = build_segments(resp, tmp_path)
    assert len(segs[0].pictures) == 1
    assert Path(segs[0].pictures[0].image_path).exists()


def test_non_figure_link_left_untouched(tmp_path):
    # A markdown link whose target is not an extracted figure id passes through as-is.
    md = "see ![diagram](https://example.com/x.png) here"
    rewritten, pics = _rewrite_page(md, [], tmp_path / "Segments" / "Segment_0000")
    assert rewritten == md
    assert pics == []


def test_pages_are_indexed_densely(tmp_path):
    # Even if the source pages are non-contiguous, segments are dense 0..N so the seam
    # merger sees a proper adjacency.
    resp = {"pages": [
        {"index": 5, "markdown": "![a](img-0.jpeg)", "images": [_img("img-0.jpeg")]},
        {"index": 9, "markdown": "![b](img-0.jpeg)", "images": [_img("img-0.jpeg")]},
    ]}
    segs = build_segments(resp, tmp_path)
    assert [s.index for s in segs] == [0, 1]
    assert all(len(s.pictures) == 1 for s in segs)
    assert all("![1]()" in s.content for s in segs)
