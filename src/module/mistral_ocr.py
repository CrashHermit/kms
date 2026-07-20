"""
Mistral OCR front-end — an API alternative to the local docling picture_extractor
plus the vision OCR stage.

The docling front-end needs a GPU (it renders pages and crops figures locally with
torch), and the OCR stage then asks a vision LLM to transcribe a whole page and
*infer* its reading order. Mistral's document OCR endpoint (``mistral-ocr-latest``)
does layout analysis, reading-order transcription, and figure extraction server-side
and returns, per page:

  - ``markdown``: the page transcribed in reading order, with each detected figure
    referenced inline as ``![<id>](<id>)`` at the spot it appears;
  - ``images``: each detected figure as a cropped image (base64) plus its bounding box.

This module turns that response into the same ``Segment`` backbone the rest of the
pipeline already consumes — ``content`` (markdown) + ``pictures`` (cropped figures on
disk) — so every downstream text stage (extractor → seam → … → entity) runs unchanged.
It needs only ``MISTRAL_API_KEY`` and outbound HTTPS; no GPU, no docling, no torch.

Reading order, duplication avoidance, and figure *placement* are all handled by
Mistral server-side, so the ingestion vision stages (image_filter, ocr) are skipped
when this front-end is used (see ``pipeline.build_graph(vision_frontend=False)``).

The markdown's figure references are rewritten from Mistral's ids to the pipeline's
positional ``![N]()`` convention (1-based, reading order), matching the picture
indices saved to disk, so the extractor and assembler treat a Mistral-sourced figure
exactly like a docling-sourced one.
"""

import base64
import os
import re
from pathlib import Path

import httpx

from .state import Picture, Segment

MISTRAL_OCR_URL = "https://api.mistral.ai/v1/ocr"
MISTRAL_OCR_MODEL = "mistral-ocr-latest"
MISTRAL_ENV_KEY = "MISTRAL_API_KEY"

# OCR of a full page can take a while; be generous. httpx reads HTTPS_PROXY and the
# CA bundle (SSL_CERT_FILE) from the environment via trust_env, exactly like the
# DSPy backends, so no explicit proxy/verify wiring is needed here.
_TIMEOUT = httpx.Timeout(300.0, connect=30.0)


class MistralOCRError(RuntimeError):
    """Raised when the Mistral OCR request fails or returns an unexpected shape."""


def _require_key() -> str:
    key = os.environ.get(MISTRAL_ENV_KEY)
    if not key:
        raise MistralOCRError(
            f"{MISTRAL_ENV_KEY} is not set. Export your Mistral API key "
            f"(e.g. `export {MISTRAL_ENV_KEY}=...`) before running the Mistral front-end."
        )
    return key


def ocr_pdf(pdf_bytes: bytes, pages: list[int] | None = None) -> dict:
    """Call the Mistral OCR endpoint on a PDF and return the raw JSON response.

    ``pages`` is an optional list of 0-based page numbers to limit the request;
    ``None`` processes the whole document.
    """
    data_url = "data:application/pdf;base64," + base64.b64encode(pdf_bytes).decode("ascii")
    payload: dict = {
        "model": MISTRAL_OCR_MODEL,
        "document": {"type": "document_url", "document_url": data_url},
        "include_image_base64": True,
    }
    if pages is not None:
        payload["pages"] = pages
    headers = {
        "Authorization": f"Bearer {_require_key()}",
        "Content-Type": "application/json",
    }
    try:
        response = httpx.post(MISTRAL_OCR_URL, json=payload, headers=headers, timeout=_TIMEOUT)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        body = exc.response.text[:500]
        raise MistralOCRError(
            f"Mistral OCR returned HTTP {exc.response.status_code}: {body}"
        ) from exc
    except httpx.HTTPError as exc:
        raise MistralOCRError(f"Mistral OCR request failed: {exc}") from exc
    return response.json()


# A markdown image reference: ![alt](target). Mistral sets `target` to a returned
# image id (e.g. `img-0.jpeg`); non-figure links are left untouched.
_IMG_REF = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def _write_image(data: str, path: Path) -> None:
    """Decode a base64 (optionally data-URL-prefixed) image to disk. Never raises —
    a malformed figure must not abort the whole document."""
    if not data:
        return
    if data.startswith("data:"):
        data = data.split(",", 1)[-1]
    try:
        path.write_bytes(base64.b64decode(data))
    except Exception:
        pass


def _rewrite_page(markdown: str, images: list[dict], seg_dir: Path) -> tuple[str, list[Picture]]:
    """Save a page's figures to disk and rewrite its markdown refs to ``![N]()``.

    Each figure is assigned a 1-based index by the order its ref first appears in the
    markdown (reading order); every ref is rewritten to the positional placeholder,
    and the figure is saved under that index. A figure present in ``images`` but never
    referenced inline is still saved (appended after the referenced ones) so no
    extracted figure is silently dropped. Returns the rewritten markdown and the
    ordered pictures.
    """
    by_id = {img.get("id"): img for img in images if img.get("id")}
    pictures_dir = seg_dir / "Images"
    pictures_dir.mkdir(parents=True, exist_ok=True)

    order: list[str] = []  # image ids in reading order; position + 1 == placeholder index

    def index_of(image_id: str) -> int:
        if image_id not in order:
            order.append(image_id)
        return order.index(image_id) + 1

    def repl(match: re.Match) -> str:
        target = match.group(1)
        if target not in by_id:
            return match.group(0)  # not a figure we extracted — leave the link as-is
        return f"![{index_of(target)}]()"

    rewritten = _IMG_REF.sub(repl, markdown)

    # Extracted figures the markdown never referenced still get an index (and a file).
    for image_id in by_id:
        index_of(image_id)

    pictures: list[Picture] = []
    for position, image_id in enumerate(order, start=1):
        path = pictures_dir / f"Image_{position - 1:03d}.png"
        _write_image(by_id[image_id].get("image_base64", ""), path)
        pictures.append(Picture(index=position, image_path=str(path)))
    return rewritten, pictures


def build_segments(response: dict, output_dir: str | Path) -> list[Segment]:
    """Turn a Mistral OCR response into the pipeline's Segment backbone.

    Segments are indexed densely by the order pages appear in the response (so a
    contiguous request stays adjacent for the seam merger), with ``content`` and
    ``pictures`` already filled. Figures are written under
    ``<output_dir>/Segments/Segment_XXXX/Images/``. ``Segment.image_path`` points at a
    page render that Mistral does not produce; it is unused after OCR (the assembler
    resolves pictures via ``seg_index`` + ``pictures``), so it is only a nominal path.
    """
    output_dir = Path(output_dir)
    segments: list[Segment] = []
    for order_index, page in enumerate(response.get("pages", [])):
        seg_dir = output_dir / "Segments" / f"Segment_{order_index:04d}"
        markdown, pictures = _rewrite_page(
            page.get("markdown", "") or "", page.get("images", []) or [], seg_dir
        )
        segments.append(Segment(
            index=order_index,
            image_path=str(seg_dir / "Segment.png"),
            pictures=pictures,
            content=markdown,
        ))
    return segments


def extract(
    pdf_path: str | Path,
    output_dir: str | Path = "output",
    pages: list[int] | None = None,
) -> list[Segment]:
    """PDF → Mistral OCR → Segments with content + pictures. No GPU, no docling."""
    pdf_bytes = Path(pdf_path).read_bytes()
    response = ocr_pdf(pdf_bytes, pages=pages)
    return build_segments(response, output_dir)
