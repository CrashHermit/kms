"""Consumer-facing tests for transient research evidence readers."""

from __future__ import annotations

import asyncio
import io
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import pypdfium2 as pdfium
import pytest
from PIL import Image

from kms2.config.services import OCRSettings
from kms2.core.model.research.retrieval import (
    DownloadedMedia,
    ResearchDocument,
    ResearchEvidenceEntry,
)
from kms2.core.model.source_processing.formatting import FormattingOutput
from kms2.core.pdf import PDFIUM_LOCK
from kms2.core.research_text import markdown_spans
from kms2.core.windowing import InputBudgetExceeded, TokenBudget
from kms2.ocr.mistral import (
    OCRPage,
    OCRResponse,
    PageDimensions,
    materialize_ocr_artifacts,
    ocr_pdf,
)
from kms2.retrieval.download import ExternalRetrievalError
from kms2.retrieval.media import decode_image
from kms2.service.research_evidence import (
    ResearchEvidenceService,
    _render_pdf_pages,
)


class _Counter:
    def count_texts(self, texts: list[str]) -> list[int]:
        return [len(text.split()) for text in texts]


class _WebClient:
    async def extract(self, urls: list[str]):
        raise AssertionError('No network request expected')


class _MediaClient:
    def __init__(self, data: bytes, media_type: str) -> None:
        self.data = data
        self.media_type = media_type
        self.calls: list[str] = []

    async def download(self, url: str) -> DownloadedMedia:
        self.calls.append(url)
        return DownloadedMedia(
            requested_url=url,
            final_url=url + '?resolved=1',
            retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
            media_type=self.media_type,
            data=self.data,
        )


class _Formatter:
    def __init__(self) -> None:
        self.inputs: list[str] = []

    async def aforward(self, *, blocks):
        self.inputs.extend(block.content for block in blocks)
        return [
            FormattingOutput(
                position=block.position,
                formatted_content=f'normalized: {block.content}',
            )
            for block in blocks
        ]


def _service(
    tmp_path: Path,
    *,
    media_client: _MediaClient | None = None,
    formatter: _Formatter | None = None,
    research_limit: int = 1000,
) -> ResearchEvidenceService:
    counter = _Counter()
    return ResearchEvidenceService(
        web_client=_WebClient(),
        media_client=media_client or _MediaClient(b'', 'application/pdf'),
        formatter=formatter or _Formatter(),
        formatting_budget=TokenBudget(counter, 1000),
        research_budget=TokenBudget(counter, research_limit),
        ocr_settings=OCRSettings(render_scale=0.5),
        temporary_directory=tmp_path,
        passage_token_budget=8,
        formatting_batch_size=4,
        pdf_search_result_limit=12,
        max_image_pixels=100_000,
        max_download_bytes=1_000_000,
    )


def _two_page_pdf(*, native_text: bool = True) -> bytes:
    first_stream = (
        b'BT /F1 16 Tf 30 240 Td (distinctive locator phrase) Tj ET'
        if native_text
        else b'1 0 0 rg 40 40 80 140 re f'
    )
    second_stream = b'0 0 1 rg 40 40 120 160 re f'
    objects = [
        b'<< /Type /Catalog /Pages 2 0 R >>',
        b'<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>',
        b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] '
        b'/Resources << /Font << /F1 7 0 R >> >> /Contents 5 0 R >>',
        b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] '
        b'/Resources << >> /Contents 6 0 R >>',
        b'<< /Length '
        + str(len(first_stream)).encode()
        + b' >>\nstream\n'
        + first_stream
        + b'\nendstream',
        b'<< /Length '
        + str(len(second_stream)).encode()
        + b' >>\nstream\n'
        + second_stream
        + b'\nendstream',
        b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>',
    ]
    document = bytearray(b'%PDF-1.4\n')
    offsets = [0]
    for index, body in enumerate(objects, start=1):
        offsets.append(len(document))
        document.extend(f'{index} 0 obj\n'.encode())
        document.extend(body)
        document.extend(b'\nendobj\n')
    xref_offset = len(document)
    document.extend(f'xref\n0 {len(objects) + 1}\n'.encode())
    document.extend(b'0000000000 65535 f \n')
    for offset in offsets[1:]:
        document.extend(f'{offset:010d} 00000 n \n'.encode())
    document.extend(
        f'trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n'
        f'startxref\n{xref_offset}\n%%EOF\n'.encode()
    )
    return bytes(document)


def test_markdown_spans_reconstruct_complete_structures() -> None:
    markdown = (
        '# Definitions\n\n'
        'The value $x^2$ has a condition.\n\n'
        '$$\n\\sum_{i=1}^{n} i\n$$\n\n'
        '| name | value |\n| --- | ---: |\n| alpha | 2 |\n\n'
        '```python\nprint("table")\n```\n\n'
        '![diagram](../img/plot.svg) and [PDF](book.pdf)\n'
        '[term]: https://example.org/definition\n'
    )
    spans = markdown_spans(markdown, _Counter(), passage_token_budget=12)

    assert ''.join(span.raw_content for span in spans) == markdown
    assert all(
        markdown.splitlines(keepends=True)[span.line_start : span.line_end]
        for span in spans
    )
    assert any('\\sum_{i=1}^{n}' in span.raw_content for span in spans)
    assert any('| alpha | 2 |' in span.raw_content for span in spans)
    assert any('print("table")' in span.raw_content for span in spans)
    assert any('book.pdf' in span.raw_content for span in spans)


def test_indivisible_markdown_span_is_never_truncated(
    tmp_path: Path,
) -> None:
    long_math = '$$\n' + 'x ' * 20 + '\n$$\n'
    spans = markdown_spans(long_math, _Counter(), passage_token_budget=2)
    assert len(spans) == 1
    assert spans[0].raw_content == long_math

    async def scenario() -> None:
        formatter = _Formatter()
        service = _service(tmp_path, formatter=formatter, research_limit=2)
        entry = ResearchEvidenceEntry(
            document_position=0,
            kind='web_passage',
            url='https://example.org/page',
            retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
            raw_content=spans[0].raw_content,
        )
        with pytest.raises(InputBudgetExceeded):
            await service.read_evidence([entry])
        assert formatter.inputs == []

    asyncio.run(scenario())


def test_web_discovery_and_read_keep_raw_and_formatted_content(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        formatter = _Formatter()
        service = _service(tmp_path, formatter=formatter)
        document = ResearchDocument(
            url='https://example.org/topic/page',
            title='Provided title',
            retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
            media_type='text/markdown',
            raw_markdown='A complete passage with math $x^2$.\n',
            image_urls=['../plot.png'],
        )
        discovered = await service.discover_web(document, document_position=4)
        passage, image = discovered
        assert passage.document_position == 4
        assert passage.preview == passage.raw_content[:160]
        assert image.url == 'https://example.org/plot.png'
        assert not image.inspected

        read = await service.read_evidence([passage])
        assert read[0].raw_content == passage.raw_content
        assert read[0].content == f'normalized: {passage.raw_content}'
        assert formatter.inputs == [passage.raw_content]

    asyncio.run(scenario())


def test_image_decoder_handles_raster_and_static_svg(tmp_path: Path) -> None:
    image = Image.new('RGB', (40, 30), 'white')
    image.putpixel((5, 6), (255, 0, 0))
    source = io.BytesIO()
    image.save(source, format='PNG')
    png = source.getvalue()
    raster = DownloadedMedia(
        requested_url='https://example.org/plot.png',
        final_url='https://example.org/plot.png',
        retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
        media_type='image/png',
        data=png,
    )
    native_png, mime_type, original, original_type, visual_note = decode_image(
        raster,
        max_pixels=10_000,
        max_bytes=100_000,
    )
    assert visual_note is None
    assert original == png
    assert original_type == 'image/png'
    with Image.open(io.BytesIO(native_png)) as decoded:
        assert decoded.size == (40, 30)
        assert decoded.getpixel((5, 6))[:3] == (255, 0, 0)

    async def read_image() -> ResearchEvidenceEntry:
        service = _service(
            tmp_path,
            media_client=_MediaClient(png, 'image/png'),
        )
        selected = ResearchEvidenceEntry(
            document_position=0,
            position=2,
            kind='image',
            url='https://example.org/plot.png',
            retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        return (await service.read_evidence([selected]))[0]

    evidence = asyncio.run(read_image())
    assert evidence.media_bytes == native_png
    assert evidence.original_media_bytes == png
    assert evidence.media_type == 'image/png'
    assert not evidence.inspected

    svg_data = (
        b'<svg xmlns="http://www.w3.org/2000/svg" width="80" height="40">'
        b'<rect width="80" height="40" fill="white"/>'
        b'<text x="4" y="25">x^2</text></svg>'
    )
    svg = raster.model_copy(
        update={'media_type': 'image/svg+xml', 'data': svg_data}
    )
    rendered, rendered_type, original_svg, original_svg_type, visual_note = (
        decode_image(
            svg,
            max_pixels=10_000,
            max_bytes=100_000,
        )
    )
    assert visual_note is None
    assert original_svg == svg_data
    assert original_svg_type == 'image/svg+xml'
    with Image.open(io.BytesIO(rendered)) as decoded:
        assert decoded.size == (80, 40)


def test_image_decoder_rejects_external_svg_and_pixel_overflow() -> None:
    base = DownloadedMedia(
        requested_url='https://example.org/plot.svg',
        final_url='https://example.org/plot.svg',
        retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
        media_type='image/svg+xml',
        data=(
            b'<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20">'
            b'<image href="file:///etc/passwd"/></svg>'
        ),
    )
    with pytest.raises(ExternalRetrievalError):
        decode_image(base, max_pixels=10_000, max_bytes=100_000)
    with pytest.raises(ExternalRetrievalError):
        decode_image(
            base.model_copy(
                update={
                    'data': (
                        b'<svg xmlns="http://www.w3.org/2000/svg" '
                        b'width="200" height="200"/>'
                    )
                }
            ),
            max_pixels=10_000,
            max_bytes=100_000,
        )


def test_pdf_open_search_and_selected_page_ocr(
    tmp_path: Path, monkeypatch
) -> None:
    native_pdf = _two_page_pdf()
    image_pdf = _two_page_pdf(native_text=False)
    service = _service(
        tmp_path,
        media_client=_MediaClient(native_pdf, 'application/pdf'),
    )
    image_service = _service(
        tmp_path,
        media_client=_MediaClient(image_pdf, 'application/pdf'),
    )
    ocr_calls: list[list[int] | None] = []

    def fake_ocr(data: bytes, settings: OCRSettings, *, pages: list[int]):
        assert data in {native_pdf, image_pdf}
        ocr_calls.append(pages)
        return OCRResponse(
            pages=[
                OCRPage(
                    index=pages[0],
                    markdown='# Selected page\n\nOCR on selected page.\n',
                    dimensions=PageDimensions(width=300, height=300),
                )
            ]
        )

    monkeypatch.setattr('kms2.service.research_evidence.ocr_pdf', fake_ocr)

    async def scenario() -> None:
        native_document = await service.open_pdf(
            'https://example.org/native.pdf'
        )
        assert native_document.page_count == 2
        assert native_document.title is None
        assert ocr_calls == []
        located = await service.search_pdf(
            native_document,
            'distinctive locator phrase',
        )
        assert located.matches[0].page_index == 0
        assert 'distinctive locator phrase' in located.matches[0].snippet
        assert ocr_calls == []

        image_document = await image_service.open_pdf(
            'https://example.org/scanned.pdf'
        )
        assert image_document.page_count == 2
        assert ocr_calls == []
        native_text = await image_service.search_pdf(
            image_document,
            'distinctive locator phrase',
        )
        assert native_text.matches == []
        assert ocr_calls == []

        page_entries = await image_service.prepare_pdf_pages(
            image_document,
            [1],
        )
        assert ocr_calls == [[1]]
        assert len(page_entries) == 1
        page = page_entries[0]
        assert page.page_index == 1
        assert page.document_position == 0
        assert page.raw_content == (
            '# Selected page\n\nOCR on selected page.\n'
        )
        assert page.content == f'normalized: {page.raw_content}'
        assert page.heading_path == ['Selected page']
        assert page.token_count > 0
        assert page.preview == page.raw_content[:160]
        assert page.media_type == 'image/png'
        assert page.media_bytes
        with Image.open(io.BytesIO(page.media_bytes)) as rendered:
            assert rendered.size == (150, 150)
        first_page, second_page = await asyncio.gather(
            image_service.prepare_pdf_pages(image_document, [0]),
            image_service.prepare_pdf_pages(image_document, [1]),
        )
        assert first_page[0].page_index == 0
        assert second_page[0].page_index == 1
        assert first_page[0].media_bytes != second_page[0].media_bytes
        assert sorted(ocr_calls) == [[0], [1], [1]]

    asyncio.run(scenario())


def test_pdfium_lock_serializes_source_and_research_readers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pdf_path = tmp_path / 'concurrent.pdf'
    pdf_bytes = _two_page_pdf()
    pdf_path.write_bytes(pdf_bytes)
    original_pdf_document = pdfium.PdfDocument
    guard = threading.Lock()
    active_documents = 0
    max_active_documents = 0

    class TrackingDocument:
        def __init__(self, path: str | Path) -> None:
            nonlocal active_documents, max_active_documents
            with guard:
                active_documents += 1
                max_active_documents = max(
                    max_active_documents, active_documents
                )
            time.sleep(0.01)
            self.document = original_pdf_document(path)

        def __len__(self) -> int:
            return len(self.document)

        def __getitem__(self, index: int):
            return self.document[index]

        def __getattr__(self, name: str):
            return getattr(self.document, name)

        def close(self) -> None:
            nonlocal active_documents
            try:
                self.document.close()
            finally:
                with guard:
                    active_documents -= 1

    monkeypatch.setattr(pdfium, 'PdfDocument', TrackingDocument)
    source_response = OCRResponse(
        pages=[
            OCRPage(
                index=1,
                markdown='',
                dimensions=PageDimensions(width=300, height=300),
            )
        ]
    )

    async def read_concurrently() -> None:
        first_render, second_render, _ = await asyncio.gather(
            asyncio.to_thread(
                _render_pdf_pages, pdf_path, [0], 0.5, 100_000, 1_000_000
            ),
            asyncio.to_thread(
                _render_pdf_pages, pdf_path, [1], 0.5, 100_000, 1_000_000
            ),
            asyncio.to_thread(
                materialize_ocr_artifacts,
                source_response,
                pdf_path,
                tmp_path / 'source-artifacts',
                0.5,
                1.0,
            ),
        )
        assert first_render[0] != second_render[1]

    asyncio.run(read_concurrently())
    assert max_active_documents == 1
    assert active_documents == 0

    lock_entered = threading.Event()
    release_lock = threading.Event()

    def hold_lock() -> None:
        with PDFIUM_LOCK:
            lock_entered.set()
            release_lock.wait()

    holder = threading.Thread(target=hold_lock)
    holder.start()
    assert lock_entered.wait(timeout=2)
    request_completed = threading.Event()

    def fake_request(request, settings):
        assert lock_entered.is_set()
        request_completed.set()
        return OCRResponse()

    monkeypatch.setattr('kms2.ocr.mistral._request_ocr', fake_request)

    async def request_while_locked() -> bool:
        task = asyncio.create_task(
            asyncio.to_thread(ocr_pdf, pdf_bytes, OCRSettings())
        )
        await asyncio.sleep(0.05)
        completed_before_unlock = request_completed.is_set()
        release_lock.set()
        await task
        return completed_before_unlock

    try:
        assert asyncio.run(request_while_locked())
    finally:
        release_lock.set()
        holder.join(timeout=2)
