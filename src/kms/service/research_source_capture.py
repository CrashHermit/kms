"""Capture public web pages and media as complete PDF snapshots."""

from __future__ import annotations

import asyncio
import io
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

import httpx
from PIL import Image, PngImagePlugin

from kms.core.model.source_processing.admission import CapturedPDFSource
from kms.retrieval.download import ExternalRetrievalError, PublicMediaClient

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


class ResearchSourceCaptureService:
    """Convert public PDFs and rendered pages to PDF snapshots."""

    def __init__(
        self,
        *,
        media_client: PublicMediaClient,
        gotenberg_client: httpx.AsyncClient,
        max_concurrency: int = 4,
    ) -> None:
        self._media_client = media_client
        self._gotenberg = GotenbergRenderer(gotenberg_client)
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def capture_source(
        self, url: str, *, title: str | None = None
    ) -> CapturedPDFSource:
        """Capture a public PDF or renderable page as a PDF snapshot."""
        async with self._semaphore:
            media = await self._media_client.capture(url)
            if media.media_type == 'application/pdf':
                data = media.data
                retrieved_at = media.retrieved_at
            else:
                screenshot = await self._gotenberg.screenshot(media.final_url)
                retrieved_at = datetime.now(UTC)
                data = await _run_in_thread(_screenshot_pdf, screenshot)
            return CapturedPDFSource(
                capture_id=uuid.uuid4().hex,
                url=media.requested_url,
                resolved_url=media.final_url,
                title=title,
                retrieved_at=retrieved_at,
                data=data,
            )


async def _run_in_thread[T](function: Callable[..., T], *args: object) -> T:
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        raise


def _screenshot_pdf(screenshot: bytes) -> bytes:
    if len(screenshot) < 24 or not screenshot.startswith(b'\x89PNG\r\n\x1a\n'):
        raise ExternalRetrievalError('Gotenberg screenshot is not a valid PNG')
    width = int.from_bytes(screenshot[16:20], 'big')
    height = int.from_bytes(screenshot[20:24], 'big')
    if width == 0 or height == 0:
        raise ExternalRetrievalError(
            'Gotenberg screenshot has invalid dimensions'
        )

    # Avoid Image.open's global decompression-bomb cap for screenshots.
    try:
        with PngImagePlugin.PngImageFile(io.BytesIO(screenshot)) as image:
            if image.size != (width, height):
                raise ExternalRetrievalError(
                    'Gotenberg screenshot has invalid dimensions'
                )
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
    except ExternalRetrievalError:
        raise
    except (OSError, SyntaxError, ValueError) as error:
        raise ExternalRetrievalError(
            'Gotenberg screenshot is not a valid PNG'
        ) from error
    try:
        return _pdf_bytes(pages)
    finally:
        for page in pages:
            page.close()


def _pdf_bytes(pages: list[Image.Image]) -> bytes:
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
