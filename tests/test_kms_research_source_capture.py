"""Tests for complete research source capture as PDF snapshots."""

import asyncio
from datetime import UTC, datetime
from io import BytesIO

import httpx
import pypdfium2
import pytest
from PIL import Image

from kms.core.model.research.retrieval import DownloadedMedia
from kms.core.model.source_processing.admission import CapturedPDFSource
from kms.retrieval.download import ExternalRetrievalError
from kms.service.research_source_capture import (
    ResearchSourceCaptureService,
    _run_in_thread,
    _screenshot_pdf,
)

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_PAGE_URL = 'https://origin.example/chapter'


def _png(width: int, height: int, color: str) -> bytes:
    output = BytesIO()
    image = Image.new('RGB', (width, height), color)
    image.save(output, format='PNG')
    image.close()
    return output.getvalue()


class _MediaClient:
    def __init__(self, captures: list[DownloadedMedia]) -> None:
        self.captures = list(captures)
        self.capture_calls: list[str] = []

    async def capture(self, url: str) -> DownloadedMedia:
        self.capture_calls.append(url)
        return self.captures.pop(0)


def _media(
    url: str,
    media_type: str,
    data: bytes | None,
    *,
    final_url: str | None = None,
) -> DownloadedMedia:
    return DownloadedMedia(
        requested_url=url,
        final_url=final_url or url,
        retrieved_at=_NOW,
        media_type=media_type,
        data=data,
    )


def _service(
    media: _MediaClient,
    transport: httpx.AsyncBaseTransport | None = None,
) -> tuple[ResearchSourceCaptureService, httpx.AsyncClient]:
    client = httpx.AsyncClient(
        base_url='http://gotenberg',
        transport=transport,
    )
    service = ResearchSourceCaptureService(
        media_client=media,
        gotenberg_client=client,
    )
    return service, client


async def _capture_and_close(
    service: ResearchSourceCaptureService,
    client: httpx.AsyncClient,
    url: str,
    *,
    title: str | None = None,
) -> CapturedPDFSource:
    try:
        return await service.capture_source(url, title=title)
    finally:
        await client.aclose()


def _pdf_page_count(pdf: bytes) -> int:
    document = pypdfium2.PdfDocument(pdf)
    try:
        return len(document)
    finally:
        document.close()


def _pdf_page_color(pdf: bytes, index: int) -> tuple[int, int, int]:
    document = pypdfium2.PdfDocument(pdf)
    try:
        image = document[index].render(scale=0.05).to_pil()
        try:
            return image.getpixel((image.width // 2, image.height // 2))[:3]
        finally:
            image.close()
    finally:
        document.close()


def test_html_capture_requests_gotenberg_and_slices_pages_in_order():
    screenshot = _png(1280, 3200, 'red')
    with Image.open(BytesIO(screenshot)) as image:
        image.paste('blue', (0, 1600, 1280, 3200))
        output = BytesIO()
        image.save(output, format='PNG')
        screenshot = output.getvalue()
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=screenshot)

    requested_url = 'https://origin.example/redirect'
    resolved_url = 'https://origin.example/chapter'
    media = _MediaClient(
        [_media(requested_url, 'text/html', None, final_url=resolved_url)]
    )
    service, client = _service(media, httpx.MockTransport(handler))

    capture = asyncio.run(
        _capture_and_close(
            service,
            client,
            requested_url,
            title='Chapter',
        )
    )

    assert isinstance(capture, CapturedPDFSource)
    assert capture.data.startswith(b'%PDF-')
    assert capture.url == requested_url
    assert capture.resolved_url == resolved_url
    assert capture.title == 'Chapter'
    assert capture.retrieved_at >= _NOW
    assert capture.capture_id
    assert media.capture_calls == [requested_url]
    assert len(requests) == 1
    request = requests[0]
    assert request.method == 'POST'
    assert request.url.path == '/forms/chromium/screenshot/url'
    assert request.headers['content-type'].startswith('multipart/form-data;')
    for field in (
        resolved_url,
        'width',
        '1280',
        'height',
        '800',
        'deviceScaleFactor',
        '1',
        'format',
        'png',
    ):
        assert field.encode() in request.content
    assert _pdf_page_count(capture.data) == 2
    assert _pdf_page_color(capture.data, 0)[0] > 200
    assert _pdf_page_color(capture.data, 1)[2] > 200


def test_gotenberg_http_error_becomes_external_retrieval_error():
    media = _MediaClient([_media(_PAGE_URL, 'text/html', b'<html></html>')])
    service, client = _service(
        media,
        httpx.MockTransport(
            lambda request: httpx.Response(502, request=request)
        ),
    )

    with pytest.raises(ExternalRetrievalError) as error:
        asyncio.run(_capture_and_close(service, client, _PAGE_URL))

    assert error.value.status_code == 502


def test_gotenberg_transport_error_becomes_external_retrieval_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError('offline', request=request)

    media = _MediaClient([_media(_PAGE_URL, 'text/html', b'<html></html>')])
    service, client = _service(media, httpx.MockTransport(handler))

    with pytest.raises(
        ExternalRetrievalError,
        match='HTML screenshot retrieval failed',
    ):
        asyncio.run(_capture_and_close(service, client, _PAGE_URL))


def test_invalid_gotenberg_screenshot_becomes_external_retrieval_error():
    media = _MediaClient([_media(_PAGE_URL, 'text/html', b'<html></html>')])
    service, client = _service(
        media,
        httpx.MockTransport(
            lambda request: httpx.Response(200, content=b'bad')
        ),
    )

    with pytest.raises(ExternalRetrievalError, match='valid PNG'):
        asyncio.run(_capture_and_close(service, client, _PAGE_URL))


def test_thread_conversion_finishes_before_cancellation_propagates(monkeypatch):
    async def exercise():
        started = asyncio.Event()
        release = asyncio.Event()

        async def delayed_to_thread(function, *args):
            started.set()
            await release.wait()
            return function(*args)

        monkeypatch.setattr(asyncio, 'to_thread', delayed_to_thread)
        conversion = asyncio.create_task(_run_in_thread(lambda: 'done'))
        await started.wait()
        conversion.cancel()
        await asyncio.sleep(0)
        assert not conversion.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await conversion

    asyncio.run(exercise())


def test_image_url_uses_the_same_screenshot_path_as_html():
    requested_url = 'https://origin.example/image.png'
    screenshot = _png(4, 5, 'green')
    media = _MediaClient([_media(requested_url, 'image/png', None)])
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=screenshot)

    service, client = _service(media, httpx.MockTransport(handler))
    capture = asyncio.run(_capture_and_close(service, client, requested_url))

    assert capture.data.startswith(b'%PDF-')
    assert capture.url == requested_url
    assert _pdf_page_count(capture.data) == 1
    assert len(requests) == 1
    assert requested_url.encode() in requests[0].content


def test_original_pdf_passes_through_unchanged():
    requested_url = 'https://origin.example/document.pdf'
    original_pdf = b'%PDF-1.7\noriginal document bytes'
    media = _MediaClient(
        [
            _media(
                requested_url,
                'application/pdf',
                original_pdf,
                final_url='https://cdn.example/document.pdf',
            )
        ]
    )
    service, client = _service(media)

    capture = asyncio.run(_capture_and_close(service, client, requested_url))

    assert capture.data == original_pdf
    assert capture.url == requested_url
    assert capture.resolved_url == 'https://cdn.example/document.pdf'


def test_screenshot_png_conversion_does_not_use_direct_image_pixel_limit():
    pdf = _screenshot_pdf(_png(1280, 1600, 'orange'))

    assert pdf.startswith(b'%PDF-')
    assert _pdf_page_count(pdf) == 1
