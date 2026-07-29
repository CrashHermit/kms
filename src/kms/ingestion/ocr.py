"""
Mistral OCR front-end — the pipeline's page ingestion, over an API (no GPU).

Mistral's document OCR endpoint (``mistral-ocr-latest``) does layout analysis,
reading-order transcription, and figure extraction server-side and returns, per
page:

  - ``markdown``: the page transcribed in reading order, with each detected
    figure referenced inline as ``![<id>](<id>)`` at the spot it appears;
  - ``images``: each detected figure as a cropped image (base64) plus its
    bounding box.

This module turns that response into the ``models.Segment`` backbone the rest
of the pipeline consumes — ``content`` (markdown) + ``pictures`` (cropped
figures on disk) — so every downstream stage (corrector → extractor → seam →
… → entity) runs on it directly. It needs only ``MISTRAL_API_KEY`` and
outbound HTTPS; no GPU, no torch.

Because reading order, duplication avoidance, and figure placement are handled
by Mistral, the correction pass (see ``corrector``) is the next stage: it
proofreads each transcription against its page image before the extractor
parses it.

The markdown's figure references are rewritten from Mistral's ids to the
pipeline's positional ``![N]()`` convention (1-based, reading order), matching
the picture indices saved to disk, so the extractor and assembler resolve them
like any other figure.
"""

import base64
import os
import re
from pathlib import Path

import httpx

from kms.core import models

MISTRAL_OCR_URL = 'https://api.mistral.ai/v1/ocr'
MISTRAL_OCR_MODEL = 'mistral-ocr-latest'
MISTRAL_ENV_KEY = 'MISTRAL_API_KEY'

# Page-render resolution for the correction pass — the scale the corrector was
# validated on (see corrector.py). Higher is sharper but heavier; 2.5 was
# sufficient.
RENDER_SCALE = 2.5

# OCR of a full page can take a while; be generous. httpx reads HTTPS_PROXY and
# the CA bundle (SSL_CERT_FILE) from the environment via trust_env, exactly like
# the DSPy backends, so no explicit proxy/verify wiring is needed here.
_TIMEOUT = httpx.Timeout(300.0, connect=30.0)


class MistralOCRError(RuntimeError):
    """Raised when the OCR request fails or returns an unexpected shape."""


def _require_key() -> str:
    """The Mistral API key, read from the environment.

    Prefers MISTRAL_API_KEY (the documented name in .env.example) and falls
    back to MISTRAL_OCR_API, the name this project's hosted environment injects
    the secret under, so the front-end runs out-of-the-box in either place.

    Returns:
        The key.

    Raises:
        MistralOCRError: If neither variable is set.
    """
    key = os.environ.get(MISTRAL_ENV_KEY) or os.environ.get('MISTRAL_OCR_API')
    if not key:
        raise MistralOCRError(
            f'{MISTRAL_ENV_KEY} is not set. Export your Mistral API key '
            f'(e.g. `export {MISTRAL_ENV_KEY}=...`) before running the '
            f'Mistral front-end.'
        )
    return key


def ocr_pdf(pdf_bytes: bytes, pages: list[int] | None = None) -> dict:
    """Call the Mistral OCR endpoint on a PDF.

    Args:
        pdf_bytes: The PDF's raw bytes.
        pages: 0-based page numbers to limit the request to. None processes
            the whole document.

    Returns:
        The raw JSON response.

    Raises:
        MistralOCRError: If the request fails or returns a non-2xx status.
    """
    data_url = 'data:application/pdf;base64,' + base64.b64encode(
        pdf_bytes
    ).decode('ascii')
    payload: dict = {
        'model': MISTRAL_OCR_MODEL,
        'document': {'type': 'document_url', 'document_url': data_url},
        'include_image_base64': True,
        # Split running heads / footers (page numbers, chapter running titles)
        # into the response's separate `header`/`footer` fields instead of
        # leaving them inline in the page markdown. We only read `markdown`, so
        # this drops page chrome from the node stream — otherwise a running
        # head can land mid-entity and split it. Needs OCR 2512+
        # (mistral-ocr-latest resolves to that).
        'extract_header': True,
        'extract_footer': True,
    }
    if pages is not None:
        payload['pages'] = pages
    headers = {
        'Authorization': f'Bearer {_require_key()}',
        'Content-Type': 'application/json',
    }
    try:
        response = httpx.post(
            MISTRAL_OCR_URL, json=payload, headers=headers, timeout=_TIMEOUT
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        body = exc.response.text[:500]
        raise MistralOCRError(
            f'Mistral OCR returned HTTP {exc.response.status_code}: {body}'
        ) from exc
    except httpx.HTTPError as exc:
        raise MistralOCRError(f'Mistral OCR request failed: {exc}') from exc
    return response.json()


# A markdown image reference: ![alt](target). Mistral sets `target` to a
# returned image id (e.g. `img-0.jpeg`); non-figure links are left untouched.
_IMG_REF = re.compile(r'!\[[^\]]*\]\(([^)]+)\)')


def _write_image(data: str, path: Path) -> None:
    """Decode a base64 (optionally data-URL-prefixed) image to disk.

    Never raises — a malformed figure must not abort the whole document.

    Args:
        data: The base64 payload, with or without a data-URL prefix.
        path: Where to write the decoded bytes.
    """
    if not data:
        return
    if data.startswith('data:'):
        data = data.split(',', 1)[-1]
    try:
        path.write_bytes(base64.b64decode(data))
    except (ValueError, OSError):
        # A malformed figure (bad base64 -> ValueError) or a write failure
        # (OSError) must not abort the whole document — skip just this figure.
        pass


def _rewrite_page(
    markdown: str, images: list[dict], segment_dir: Path
) -> tuple[str, list[models.Picture]]:
    """Save a page's figures to disk and rewrite its refs to ``![N]()``.

    Each figure is assigned a 1-based index by the order its ref first appears
    in the markdown (reading order); every ref is rewritten to the positional
    placeholder, and the figure is saved under that index. A figure present in
    ``images`` but never referenced inline is still saved (appended after the
    referenced ones) so no extracted figure is silently dropped.

    Args:
        markdown: The page's transcription, with Mistral's figure ids.
        images: The page's extracted figures, each with an ``id``.
        segment_dir: The page's output directory.

    Returns:
        The rewritten markdown and the ordered pictures.
    """
    images_by_id = {
        image.get('id'): image for image in images if image.get('id')
    }
    pictures_dir = segment_dir / 'Images'
    pictures_dir.mkdir(parents=True, exist_ok=True)

    # Image ids in reading order; position + 1 == placeholder index.
    order: list[str] = []

    def index_of(image_id: str) -> int:
        """The 1-based placeholder index, assigned on first sight."""
        if image_id not in order:
            order.append(image_id)
        return order.index(image_id) + 1

    def replace(match: re.Match) -> str:
        """Rewrite one markdown image link to its ``![N]()`` form."""
        target = match.group(1)
        if target not in images_by_id:
            # Not a figure we extracted — leave the link as-is.
            return match.group(0)
        return f'![{index_of(target)}]()'

    rewritten = _IMG_REF.sub(replace, markdown)

    # Extracted figures the markdown never referenced still get an index (and
    # a file).
    for image_id in images_by_id:
        index_of(image_id)

    pictures: list[models.Picture] = []
    for position, image_id in enumerate(order, start=1):
        path = pictures_dir / f'Image_{position - 1:03d}.png'
        _write_image(images_by_id[image_id].get('image_base64', ''), path)
        pictures.append(models.Picture(index=position, image_path=str(path)))
    return rewritten, pictures


def build_segments(
    response: dict, output_dir: str | Path
) -> list[models.Segment]:
    """Turn a Mistral OCR response into the ``models.Segment`` backbone.

    Segments are indexed densely by the order pages appear in the response (so
    a contiguous request stays adjacent for the seam merger), with ``content``
    and ``pictures`` already filled. Figures are written under
    ``<output_dir>/Segments/Segment_XXXX/Images/``.
    ``models.Segment.image_path`` points at a page render that Mistral does not
    produce; it is unused after OCR (the assembler resolves pictures via
    ``segment_index`` + ``pictures``), so it is only a nominal path.

    Args:
        response: The raw OCR JSON.
        output_dir: Root directory the per-page assets are written under.

    Returns:
        The ordered segment backbone.
    """
    output_dir = Path(output_dir)
    segments: list[models.Segment] = []
    for order_index, page in enumerate(response.get('pages', [])):
        segment_dir = output_dir / 'Segments' / f'Segment_{order_index:04d}'
        markdown, pictures = _rewrite_page(
            page.get('markdown', '') or '',
            page.get('images', []) or [],
            segment_dir,
        )
        segments.append(
            models.Segment(
                index=order_index,
                image_path=str(segment_dir / 'Segment.png'),
                pictures=pictures,
                content=markdown,
            )
        )
    return segments


def _render_page_images(
    pdf_path: str | Path,
    segments: list[models.Segment],
    pages: list[int] | None,
) -> None:
    """Rasterize each segment's source page to its ``Segment.png``.

    Mistral does not return a full-page render (only figure crops), but the
    downstream correction pass needs the page image to check the transcription
    against. Render it here — CPU-only, no GPU — at the same scale the
    corrector was validated on. The source page for segment ``i`` is
    ``pages[i]`` when a subset was requested, else ``i`` (a whole-document
    request returns pages densely in order). pypdfium2 is an optional dep (the
    ``mistral`` extra); its import is deferred so the light core stays
    installable.

    Args:
        pdf_path: The source PDF.
        segments: The segment backbone, in response order.
        pages: The 0-based pages that were requested, or None for all.

    Raises:
        MistralOCRError: If pypdfium2 is not installed.
    """
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise MistralOCRError(
            'pypdfium2 is required to render page images for the correction '
            'pass. Install the Mistral front-end deps: '
            ' uv sync --extra mistral'
        ) from exc

    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        for i, segment in enumerate(segments):
            source_page = pages[i] if pages is not None else i
            image = pdf[source_page].render(scale=RENDER_SCALE).to_pil()
            Path(segment.image_path).parent.mkdir(parents=True, exist_ok=True)
            image.save(segment.image_path)
    finally:
        pdf.close()


def extract(
    pdf_path: str | Path,
    output_dir: str | Path = 'output',
    pages: list[int] | None = None,
    render_pages: bool = True,
) -> list[models.Segment]:
    """PDF → Mistral OCR → Segments with content + pictures.

    No GPU, no docling.

    Args:
        pdf_path: The source PDF.
        output_dir: Root directory the per-page assets are written under.
        pages: 0-based pages to limit the request to, or None for all.
        render_pages: When True (default) each page is also rasterized to its
            ``Segment.png`` so the correction pass has an image to proofread
            against. Pass False to skip rendering (e.g. text-only runs with no
            correction pass).

    Returns:
        The ordered segment backbone.
    """
    pdf_bytes = Path(pdf_path).read_bytes()
    response = ocr_pdf(pdf_bytes, pages=pages)
    segments = build_segments(response, output_dir)
    if render_pages:
        _render_page_images(pdf_path, segments, pages)
    return segments
