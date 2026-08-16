"""Mistral OCR front-end: PDF to segments with pictures and page images."""

import base64
import re
from pathlib import Path

import httpx
import pypdfium2 as pdfium

from kms import config
from kms.core import models, state

_TIMEOUT = httpx.Timeout(300.0, connect=30.0)


class MistralOCRError(RuntimeError):
    """Raised when the Mistral OCR API cannot be used or fails."""

    pass


def _require_key() -> str:
    """Returns the configured Mistral API key.

    Raises:
        MistralOCRError: If no key is configured.
    """
    key = config.get_settings().ocr.api_key
    if not key:
        raise MistralOCRError(
            'KMS_OCR__API_KEY is not set. Export your Mistral API key '
            '(e.g. `export KMS_OCR__API_KEY=...`) before running the '
            'Mistral front-end.'
        )
    return key


def ocr_pdf(pdf_bytes: bytes, pages: list[int] | None = None) -> dict:
    """Runs Mistral OCR on a PDF and returns its JSON response.

    Args:
        pdf_bytes: The PDF file bytes.
        pages: Optional 0-based page indexes to OCR.

    Returns:
        The parsed OCR response body.

    Raises:
        MistralOCRError: If the request fails or the API errors.
    """
    data_url = 'data:application/pdf;base64,' + base64.b64encode(
        pdf_bytes
    ).decode('ascii')
    payload: dict = {
        'model': config.get_settings().ocr.model,
        'document': {'type': 'document_url', 'document_url': data_url},
        'include_image_base64': True,
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
            config.get_settings().ocr.url,
            json=payload,
            headers=headers,
            timeout=_TIMEOUT,
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


_IMG_REF = re.compile(r'!\[[^\]]*\]\(([^)]+)\)')


def _write_image(data: str, path: Path) -> None:
    """Decodes base64 image data into a file, tolerating bad data."""
    if not data:
        return
    if data.startswith('data:'):
        data = data.split(',', 1)[-1]
    try:
        path.write_bytes(base64.b64decode(data))
    except (ValueError, OSError):
        pass


def _rewrite_page(
    markdown: str, images: list[dict], segment_dir: Path
) -> tuple[str, list[models.Picture]]:
    """Rewrites image placeholders to local indexes and saves pictures.

    Args:
        markdown: The page's OCR markdown.
        images: The page's image records from the OCR response.
        segment_dir: The segment's output directory.

    Returns:
        ``(markdown, pictures)`` with placeholders rewritten to
        ``![N]()`` and pictures saved under ``Images/``.
    """
    images_by_id = {
        image.get('id'): image for image in images if image.get('id')
    }
    pictures_dir = segment_dir / 'Images'
    pictures_dir.mkdir(parents=True, exist_ok=True)
    order: list[str] = []

    def index_of(image_id: str) -> int:
        """Returns the 1-based display index, assigning it on first use."""
        if image_id not in order:
            order.append(image_id)
        return order.index(image_id) + 1

    def replace(match: re.Match) -> str:
        """Rewrites one placeholder to its local ``![N]()`` form."""
        target = match.group(1)
        if target not in images_by_id:
            return match.group(0)
        return f'![{index_of(target)}]()'

    rewritten = _IMG_REF.sub(replace, markdown)
    for image_id in images_by_id:
        index_of(image_id)

    pictures: list[models.Picture] = []
    for position, image_id in enumerate(order, start=1):
        path = pictures_dir / f'Image_{position - 1:03d}.png'
        _write_image(images_by_id[image_id].get('image_base64', ''), path)
        pictures.append(models.Picture(index=position, image_path=str(path)))
    return rewritten, pictures


def _with_footer(markdown: str, footer: str | None) -> str:
    """Appends the extracted page footer to the markdown, if any."""
    footer = (footer or '').strip()
    if not footer:
        return markdown
    return f'{markdown.rstrip()}\n\n{footer}'


def build_segments(
    response: dict, output_dir: str | Path
) -> list[models.Segment]:
    """Builds one Segment per OCR page under ``Segments/``.

    Args:
        response: The Mistral OCR response body.
        output_dir: The root output directory.

    Returns:
        The ordered list of segments.
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
                content=_with_footer(markdown, page.get('footer')),
            )
        )
    return segments


def _render_page_images(
    pdf_path: str | Path,
    segments: list[models.Segment],
    pages: list[int] | None,
) -> None:
    """Renders each segment's page to PNG at its configured image path."""
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        for i, segment in enumerate(segments):
            source_page = pages[i] if pages is not None else i
            image = (
                pdf[source_page]
                .render(scale=config.get_settings().ocr.render_scale)
                .to_pil()
            )
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
    """OCR's a PDF into ordered segments with pictures and page images.

    Args:
        pdf_path: Path to the PDF file.
        output_dir: Root directory for segment output.
        pages: Optional 0-based page indexes to OCR.
        render_pages: When True, render each page to PNG.

    Returns:
        The ordered list of segments.
    """
    pdf_bytes = Path(pdf_path).read_bytes()
    response = ocr_pdf(pdf_bytes, pages=pages)
    segments = build_segments(response, output_dir)
    if render_pages:
        _render_page_images(pdf_path, segments, pages)
    return segments


class OCRNode:
    def run(self, current_state: state.State) -> dict:
        segments = extract(
            current_state['pdf_path'],
            output_dir=current_state['output_dir'],
            pages=current_state.get('pages'),
        )
        return {'segments': segments}
