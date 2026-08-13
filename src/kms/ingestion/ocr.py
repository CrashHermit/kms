import base64
import os
import re
from pathlib import Path

import httpx
import pypdfium2 as pdfium

from kms.core import models

MISTRAL_OCR_URL = 'https://api.mistral.ai/v1/ocr'
MISTRAL_OCR_MODEL = 'mistral-ocr-latest'
MISTRAL_ENV_KEY = 'MISTRAL_API_KEY'
RENDER_SCALE = 2.5
_TIMEOUT = httpx.Timeout(300.0, connect=30.0)


class MistralOCRError(RuntimeError):
    pass


def _require_key() -> str:
    key = os.environ.get(MISTRAL_ENV_KEY) or os.environ.get('MISTRAL_OCR_API')
    if not key:
        raise MistralOCRError(
            f'{MISTRAL_ENV_KEY} is not set. Export your Mistral API key '
            f'(e.g. `export {MISTRAL_ENV_KEY}=...`) before running the '
            f'Mistral front-end.'
        )
    return key


def ocr_pdf(pdf_bytes: bytes, pages: list[int] | None = None) -> dict:
    data_url = 'data:application/pdf;base64,' + base64.b64encode(
        pdf_bytes
    ).decode('ascii')
    payload: dict = {
        'model': MISTRAL_OCR_MODEL,
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


_IMG_REF = re.compile(r'!\[[^\]]*\]\(([^)]+)\)')


def _write_image(data: str, path: Path) -> None:
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
    images_by_id = {
        image.get('id'): image for image in images if image.get('id')
    }
    pictures_dir = segment_dir / 'Images'
    pictures_dir.mkdir(parents=True, exist_ok=True)
    order: list[str] = []

    def index_of(image_id: str) -> int:
        if image_id not in order:
            order.append(image_id)
        return order.index(image_id) + 1

    def replace(match: re.Match) -> str:
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
    footer = (footer or '').strip()
    if not footer:
        return markdown
    return f'{markdown.rstrip()}\n\n{footer}'


def build_segments(
    response: dict, output_dir: str | Path
) -> list[models.Segment]:
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
    pdf_bytes = Path(pdf_path).read_bytes()
    response = ocr_pdf(pdf_bytes, pages=pages)
    segments = build_segments(response, output_dir)
    if render_pages:
        _render_page_images(pdf_path, segments, pages)
    return segments
