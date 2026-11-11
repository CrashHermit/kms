"""Render public pages and convert long screenshots to PDF."""

import io

import httpx
from PIL import PngImagePlugin

from kms.source.capture.inspection import ExternalRetrievalError

_PAGE_SLICE_HEIGHT = 1600


class GotenbergRenderer:
    """Request a PNG screenshot of a public page from Gotenberg."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def screenshot(self, url: str) -> bytes:
        """Retrieve a page screenshot, translating HTTP failures at retrieval."""
        fields = {
            'url': url,
            'width': '1280',
            'height': '800',
            'deviceScaleFactor': '1',
            'format': 'png',
            'waitDelay': '2s',
        }
        try:
            response = await self._client.post(
                '/forms/chromium/screenshot/url',
                files=[(name, (None, value)) for name, value in fields.items()],
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise ExternalRetrievalError(
                'HTML screenshot retrieval failed',
                status_code=error.response.status_code,
            ) from None
        except httpx.HTTPError:
            raise ExternalRetrievalError(
                'HTML screenshot retrieval failed'
            ) from None
        return response.content


def _screenshot_pdf(screenshot: bytes) -> bytes:
    # Avoid Image.open's global decompression-bomb cap for long screenshots.
    with PngImagePlugin.PngImageFile(io.BytesIO(screenshot)) as image:
        pages = [
            image.crop(
                (
                    0,
                    top,
                    image.width,
                    min(top + _PAGE_SLICE_HEIGHT, image.height),
                )
            )
            for top in range(0, image.height, _PAGE_SLICE_HEIGHT)
        ]
    try:
        output = io.BytesIO()
        first, *remaining = pages
        first.save(
            output,
            format='PDF',
            save_all=True,
            append_images=remaining,
            resolution=96,
        )
        return output.getvalue()
    finally:
        for page in pages:
            page.close()
