"""Behavioral tests for explicit web-page capture."""

import asyncio
from io import BytesIO

import httpx
import pypdfium2
import pytest
from PIL import Image

from kms.source.capture.capture import SourceCapture
from kms.source.capture.inspection import ExternalRetrievalError
from kms.source.capture.models import PDFCandidate, URLInspection
from kms.source.capture.rendering import GotenbergRenderer, _screenshot_pdf
from kms.source.models.inputs import CapturedPDFSource

_URL = 'https://public.example/page'


def _png(width: int, height: int) -> bytes:
    output = BytesIO()
    Image.new('RGB', (width, height), 'red').save(output, format='PNG')
    return output.getvalue()


class _Inspector:
    def __init__(self, inspection: URLInspection) -> None:
        self.inspection = inspection
        self.urls: list[str] = []

    async def inspect(self, url: str) -> URLInspection:
        self.urls.append(url)
        return self.inspection


class _Renderer:
    def __init__(self, screenshot: bytes = b'') -> None:
        self.screenshot_bytes = screenshot
        self.urls: list[str] = []

    async def screenshot(self, url: str) -> bytes:
        self.urls.append(url)
        return self.screenshot_bytes


def _inspection(media_type: str, *, final_url: str = _URL) -> URLInspection:
    from datetime import UTC, datetime

    return URLInspection(
        requested_url=_URL,
        final_url=final_url,
        retrieved_at=datetime.now(UTC),
        media_type=media_type,
    )


def test_page_capture_renders_final_url_into_ordered_pdf_pages() -> None:
    async def exercise() -> CapturedPDFSource:
        inspector = _Inspector(
            _inspection('text/html', final_url='https://cdn.example/page')
        )
        screenshot = Image.open(BytesIO(_png(1280, 3200)))
        screenshot.paste('blue', (0, 1600, 1280, 3200))
        output = BytesIO()
        screenshot.save(output, format='PNG')
        renderer = _Renderer(output.getvalue())
        capture = await SourceCapture(inspector, renderer).capture(
            _URL, title='Page'
        )
        assert isinstance(capture, CapturedPDFSource)
        assert inspector.urls == [_URL]
        assert renderer.urls == ['https://cdn.example/page']
        return capture

    capture = asyncio.run(exercise())
    document = pypdfium2.PdfDocument(capture.data)
    try:
        assert len(document) == 2
        for index, channel in ((0, 0), (1, 2)):
            page = document[index].render(scale=0.05).to_pil()
            try:
                assert (
                    page.getpixel((page.width // 2, page.height // 2))[channel]
                    > 200
                )
            finally:
                page.close()
    finally:
        document.close()
    assert capture.url == _URL
    assert capture.resolved_url == 'https://cdn.example/page'
    assert capture.title == 'Page'
    assert capture.capture_id


def test_pdf_inspection_returns_final_link_without_rendering() -> None:
    inspector = _Inspector(
        _inspection('application/pdf', final_url='https://cdn.example/file.pdf')
    )
    renderer = _Renderer()
    result = asyncio.run(
        SourceCapture(inspector, renderer).capture(_URL, title='PDF')
    )
    assert result == PDFCandidate(
        url='https://cdn.example/file.pdf', title='PDF'
    )
    assert renderer.urls == []


def test_image_url_uses_the_web_page_rendering_path() -> None:
    inspector = _Inspector(_inspection('image/png'))
    renderer = _Renderer(_png(8, 8))
    result = asyncio.run(SourceCapture(inspector, renderer).capture(_URL))
    assert isinstance(result, CapturedPDFSource)
    assert renderer.urls == [_URL]


def test_gotenberg_http_errors_are_translated() -> None:
    async def exercise() -> None:
        client = httpx.AsyncClient(
            base_url='http://gotenberg',
            transport=httpx.MockTransport(
                lambda request: httpx.Response(502, request=request)
            ),
        )
        try:
            with pytest.raises(ExternalRetrievalError) as error:
                await GotenbergRenderer(client).screenshot(_URL)
            assert error.value.status_code == 502
        finally:
            await client.aclose()

    asyncio.run(exercise())


def test_gotenberg_screenshot_uses_configured_endpoint_and_fields() -> None:
    async def exercise() -> None:
        requests: list[httpx.Request] = []

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, content=b'png')

        client = httpx.AsyncClient(
            base_url='http://renderer.local:3000',
            transport=httpx.MockTransport(respond),
        )
        try:
            assert await GotenbergRenderer(client).screenshot(_URL) == b'png'
        finally:
            await client.aclose()

        assert len(requests) == 1
        request = requests[0]
        assert request.method == 'POST'
        assert request.url.path == '/forms/chromium/screenshot/url'
        assert request.headers['content-type'].startswith(
            'multipart/form-data;'
        )
        for value in (
            _URL,
            'width',
            '1280',
            'height',
            '800',
            'deviceScaleFactor',
            '1',
            'format',
            'png',
            'waitDelay',
            '2s',
        ):
            assert value.encode() in request.content

    asyncio.run(exercise())


def test_gotenberg_transport_errors_are_translated() -> None:
    async def exercise() -> None:
        def respond(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError('offline', request=request)

        client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        try:
            with pytest.raises(
                ExternalRetrievalError, match='HTML screenshot retrieval failed'
            ):
                await GotenbergRenderer(client).screenshot(_URL)
        finally:
            await client.aclose()

    asyncio.run(exercise())


def test_thread_conversion_finishes_before_cancellation_propagates(
    monkeypatch,
) -> None:
    from kms.concurrency import run_blocking_work

    async def exercise() -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        async def delayed_to_thread(function, *args):
            started.set()
            await release.wait()
            return function(*args)

        monkeypatch.setattr(asyncio, 'to_thread', delayed_to_thread)
        conversion = asyncio.create_task(run_blocking_work(lambda: 'done'))
        await started.wait()
        conversion.cancel()
        await asyncio.sleep(0)
        assert not conversion.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await conversion

    asyncio.run(exercise())


def test_long_screenshot_conversion_slices_pages() -> None:
    pdf = _screenshot_pdf(_png(1280, 3200))
    document = pypdfium2.PdfDocument(pdf)
    try:
        assert len(document) == 2
    finally:
        document.close()
