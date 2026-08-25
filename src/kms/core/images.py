"""Image loading utilities for vision-native construction stages."""

import base64
import io
from pathlib import Path

import dspy
from PIL import Image


def _resize_bytes(data: bytes, max_dim: int) -> bytes:
    """Resize image bytes so the longest side is at most max_dim."""
    with Image.open(io.BytesIO(data)) as img:
        width, height = img.size
        longest = max(width, height)
        if longest <= max_dim:
            return data
        scale = max_dim / longest
        resized = img.resize(
            (round(width * scale), round(height * scale)),
            Image.Resampling.LANCZOS,
        )
        buf = io.BytesIO()
        resized.save(buf, format='PNG')
        return buf.getvalue()


def load_image(
    path: str | None, max_dim: int | None = None
) -> dspy.Image | None:
    """Load an image file into a dspy.Image, or None if absent."""
    if not path:
        return None
    data = Path(path).read_bytes()
    if max_dim is not None:
        data = _resize_bytes(data, max_dim)
    encoded = base64.b64encode(data).decode('utf-8')
    return dspy.Image(url=f'data:image/png;base64,{encoded}')


def image_url(image: dspy.Image) -> str:
    """Convert a dspy.Image to a data URI or URL for payloads."""
    url = image.url or ''
    if url.startswith('data:') or url.startswith('http'):
        return url
    path = Path(url)
    if path.exists():
        encoded = base64.b64encode(path.read_bytes()).decode('utf-8')
        return f'data:image/png;base64,{encoded}'
    return url
