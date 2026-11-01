"""Decode untrusted research image bytes into bounded native PNG inputs."""

from __future__ import annotations

import io
import math
import re
import xml.etree.ElementTree as ET

from PIL import Image, UnidentifiedImageError

from kms2.core.model.research.retrieval import DownloadedMedia
from kms2.retrieval.download import ExternalRetrievalError

_EXTERNAL_CSS = re.compile(r'@import|@font-face', re.IGNORECASE)
_CSS_URL = re.compile(r'url\(\s*["\']?([^"\')]+)', re.IGNORECASE)


def _dimensions(
    width: float, height: float, max_pixels: int
) -> tuple[int, int]:
    if not math.isfinite(width) or not math.isfinite(height):
        raise ExternalRetrievalError('Image has invalid dimensions')
    pixel_width = round(width)
    pixel_height = round(height)
    if pixel_width < 1 or pixel_height < 1:
        raise ExternalRetrievalError('Image has invalid dimensions')
    if pixel_width * pixel_height > max_pixels:
        raise ExternalRetrievalError('Image exceeds the configured pixel limit')
    return pixel_width, pixel_height


def _svg_dimensions(root: ET.Element, max_pixels: int) -> tuple[int, int]:
    view_box = root.attrib.get('viewBox') or root.attrib.get('viewbox')
    view_size: tuple[float, float] | None = None
    if view_box:
        parts = re.split(r'[ ,]+', view_box.strip())
        if len(parts) == 4:
            try:
                view_size = (float(parts[2]), float(parts[3]))
            except ValueError as error:
                raise ExternalRetrievalError(
                    'SVG has invalid dimensions'
                ) from error
    width = root.attrib.get('width')
    height = root.attrib.get('height')
    if width is None or height is None:
        if view_size is None:
            raise ExternalRetrievalError('SVG has no bounded dimensions')
        width = width or str(view_size[0])
        height = height or str(view_size[1])
    width_value = re.fullmatch(r'\s*([0-9]+(?:\.[0-9]+)?)\s*(?:px)?\s*', width)
    height_value = re.fullmatch(
        r'\s*([0-9]+(?:\.[0-9]+)?)\s*(?:px)?\s*', height
    )
    if width_value is None or height_value is None:
        raise ExternalRetrievalError('SVG dimensions must be absolute pixels')
    return _dimensions(
        float(width_value.group(1)),
        float(height_value.group(1)),
        max_pixels,
    )


def _validate_svg(svg_bytes: bytes, max_pixels: int) -> tuple[str, int, int]:
    upper = svg_bytes.upper()
    if b'<!DOCTYPE' in upper or b'<!ENTITY' in upper:
        raise ExternalRetrievalError('SVG declarations are not allowed')
    if re.search(rb'<\?(?!xml(?:\s|\?>))', svg_bytes, re.IGNORECASE):
        raise ExternalRetrievalError(
            'SVG processing instructions are not allowed'
        )
    try:
        root = ET.fromstring(svg_bytes)
    except (ET.ParseError, UnicodeDecodeError) as error:
        raise ExternalRetrievalError('Malformed SVG image') from error
    if not root.tag.endswith('svg'):
        raise ExternalRetrievalError('Image is not an SVG document')

    for element in root.iter():
        if element.tag.rsplit('}', 1)[-1].lower() in {
            'script',
            'foreignobject',
        }:
            raise ExternalRetrievalError('Active SVG content is not allowed')
        for value in (*element.attrib.values(), element.text or ''):
            if _EXTERNAL_CSS.search(value):
                raise ExternalRetrievalError(
                    'External SVG styles are not allowed'
                )
            for match in _CSS_URL.finditer(value):
                target = match.group(1).strip()
                if not target.startswith(('#', 'data:')):
                    raise ExternalRetrievalError(
                        'External SVG resources are not allowed'
                    )
        for name, value in element.attrib.items():
            if name.rsplit('}', 1)[-1].lower() in {'href', 'src'}:
                if value and not value.startswith(('#', 'data:')):
                    raise ExternalRetrievalError(
                        'External SVG resources are not allowed'
                    )
                if value.lower().startswith('data:image/svg+xml'):
                    raise ExternalRetrievalError(
                        'Embedded SVG documents are not allowed'
                    )
    width, height = _svg_dimensions(root, max_pixels)
    try:
        return svg_bytes.decode('utf-8'), width, height
    except UnicodeDecodeError as error:
        raise ExternalRetrievalError('Malformed SVG image') from error


def _png_bytes(image: Image.Image) -> bytes:
    normalized = image.convert('RGBA')
    try:
        output = io.BytesIO()
        normalized.save(output, format='PNG')
        return output.getvalue()
    finally:
        normalized.close()


def decode_image(
    media: DownloadedMedia,
    *,
    max_pixels: int,
    max_bytes: int,
) -> tuple[bytes, str, bytes, str, str | None]:
    """Return native PNG, original bytes/MIME, and any visual-scope note."""
    if len(media.data) > max_bytes:
        raise ExternalRetrievalError('Image exceeds the configured byte limit')
    media_type = media.media_type.split(';', 1)[0].strip().lower()
    original_type = media_type or 'application/octet-stream'
    is_svg = media_type in {
        'image/svg+xml',
        'image/svg',
    } or media.data.lstrip().startswith(b'<svg')
    if is_svg:
        import resvg_py

        svg_string, width, height = _validate_svg(media.data, max_pixels)
        try:
            rendered = resvg_py.svg_to_bytes(
                svg_string=svg_string,
                width=width,
                height=height,
            )
        except (OSError, ValueError) as error:
            raise ExternalRetrievalError(
                'Malformed or unsupported SVG image'
            ) from error
        if len(rendered) > max_bytes:
            raise ExternalRetrievalError(
                'Rendered image exceeds the byte limit'
            )
        try:
            with Image.open(io.BytesIO(rendered)) as image:
                _dimensions(*image.size, max_pixels)
                image.load()
                png_bytes = _png_bytes(image)
        except (
            UnidentifiedImageError,
            OSError,
            Image.DecompressionBombError,
        ) as error:
            raise ExternalRetrievalError(
                'Malformed rendered SVG image'
            ) from error
        if len(png_bytes) > max_bytes:
            raise ExternalRetrievalError(
                'Rendered image exceeds the byte limit'
            )
        return png_bytes, 'image/png', media.data, original_type, None

    frame_note: str | None = None
    try:
        with Image.open(io.BytesIO(media.data)) as image:
            _dimensions(*image.size, max_pixels)
            if getattr(image, 'n_frames', 1) > 1:
                frame_note = (
                    'Animated image: only the first frame was inspected.'
                )
            image.seek(0)
            image.load()
            png_bytes = _png_bytes(image)
    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
    ) as error:
        raise ExternalRetrievalError(
            'Malformed or unsupported image'
        ) from error
    if len(png_bytes) > max_bytes:
        raise ExternalRetrievalError('Rendered image exceeds the byte limit')
    return png_bytes, 'image/png', media.data, original_type, frame_note
