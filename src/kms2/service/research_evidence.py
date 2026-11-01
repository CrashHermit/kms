"""Read extracted web content and selected original PDF pages."""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
from urllib.parse import urljoin
from uuid import uuid4

import pypdfium2 as pdfium
from markdown_it import MarkdownIt

from kms2.config.services import OCRSettings
from kms2.core.model.research.retrieval import (
    PdfSearchMatch,
    PdfSearchResult,
    ResearchDocument,
    ResearchEvidenceEntry,
    WebExtractionBatch,
)
from kms2.core.model.source_processing.formatting import FormattingInput
from kms2.core.pdf import PDFIUM_LOCK
from kms2.core.research_text import markdown_spans
from kms2.core.windowing import InputBudgetExceeded, TokenBudget, pack_items
from kms2.module.source_processing.formatting import FormatterModule
from kms2.ocr.mistral import materialize_ocr_artifacts, ocr_pdf
from kms2.retrieval.download import (
    ExternalRetrievalError,
    PublicMediaClient,
)
from kms2.retrieval.media import decode_image
from kms2.retrieval.tavily import TavilyWebClient


class ResearchEvidenceService:
    """Retrieve, format, and render evidence without durable side effects."""

    def __init__(
        self,
        *,
        web_client: TavilyWebClient,
        media_client: PublicMediaClient,
        formatter: FormatterModule,
        formatting_budget: TokenBudget,
        research_budget: TokenBudget,
        ocr_settings: OCRSettings,
        temporary_directory: Path,
        passage_token_budget: int,
        formatting_batch_size: int,
        pdf_search_result_limit: int,
        max_image_pixels: int,
        max_download_bytes: int,
        max_concurrency: int = 4,
    ) -> None:
        self._web_client = web_client
        self._media_client = media_client
        self._formatter = formatter
        self._formatting_budget = formatting_budget
        self._research_budget = research_budget
        self._ocr_settings = ocr_settings
        self._temporary_directory = Path(temporary_directory)
        self._passage_token_budget = passage_token_budget
        self._formatting_batch_size = formatting_batch_size
        self._pdf_search_result_limit = pdf_search_result_limit
        self._max_image_pixels = max_image_pixels
        self._max_download_bytes = max_download_bytes
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def fetch_web(self, urls: list[str]) -> WebExtractionBatch:
        """Extract selected URLs through the configured Tavily adapter."""
        async with self._semaphore:
            return await self._web_client.extract(urls)

    async def discover_web(
        self,
        document: ResearchDocument,
        document_position: int,
    ) -> list[ResearchEvidenceEntry]:
        """Build source-backed passage and image entries for extracted Markdown."""
        markdown = document.raw_markdown or ''
        spans = markdown_spans(
            markdown,
            self._research_budget.counter,
            passage_token_budget=self._passage_token_budget,
        )
        document_title = document.title or next(
            (span.heading_path[0] for span in spans if span.heading_path),
            None,
        )
        entries = [
            ResearchEvidenceEntry(
                document_position=document_position,
                kind='web_passage',
                url=document.url,
                resolved_url=document.resolved_url,
                title=document_title,
                retrieved_at=document.retrieved_at,
                line_start=span.line_start,
                line_end=span.line_end,
                raw_content=span.raw_content,
                heading_path=list(span.heading_path),
                token_count=span.token_count,
                preview=span.raw_content[:160],
            )
            for span in spans
        ]
        parser = MarkdownIt('commonmark').enable('table')
        image_urls = set(document.image_urls)
        for token in parser.parse(markdown):
            if token.type == 'inline' and token.children is not None:
                image_urls.update(
                    child.attrGet('src')
                    for child in token.children
                    if child.type == 'image' and child.attrGet('src')
                )
        for image_url in sorted(
            {urljoin(document.url, item) for item in image_urls}
        ):
            entries.append(
                ResearchEvidenceEntry(
                    document_position=document_position,
                    kind='image',
                    url=image_url,
                    title=document_title,
                    retrieved_at=document.retrieved_at,
                    token_count=0,
                    preview=image_url[:160],
                )
            )
        return entries

    async def open_pdf(self, url: str) -> ResearchDocument:
        """Download and inspect PDF metadata without invoking OCR."""
        async with self._semaphore:
            media = await self._media_client.download(url)
        if media.media_type.split(';', 1)[0].lower() != 'application/pdf':
            raise ExternalRetrievalError(
                f'Unsupported external media type: {media.media_type}'
            )
        path = self._temporary_directory / f'{uuid4().hex}.pdf'
        await asyncio.to_thread(path.write_bytes, media.data)
        try:
            page_count, title = await asyncio.to_thread(_pdf_metadata, path)
        except pdfium.PdfiumError as error:
            raise ExternalRetrievalError('Malformed external PDF') from error
        return ResearchDocument(
            url=media.requested_url,
            resolved_url=media.final_url,
            title=title,
            retrieved_at=media.retrieved_at,
            media_type='application/pdf',
            pdf_path=path,
            page_count=page_count,
        )

    async def search_pdf(
        self,
        document: ResearchDocument,
        query: str,
    ) -> PdfSearchResult:
        """Locate matching native PDF text without OCR or evidence inspection."""
        if document.pdf_path is None:
            raise ValueError('Research document is not an opened PDF')
        try:
            return await asyncio.to_thread(
                _search_pdf,
                document.pdf_path,
                query,
                self._pdf_search_result_limit,
            )
        except pdfium.PdfiumError as error:
            raise ExternalRetrievalError(
                'Unable to read external PDF text'
            ) from error

    async def prepare_pdf_pages(
        self,
        document: ResearchDocument,
        page_indexes: list[int],
    ) -> list[ResearchEvidenceEntry]:
        """OCR and render only explicitly selected original PDF pages."""
        if document.pdf_path is None:
            raise ValueError('Research document is not an opened PDF')
        selected = sorted(set(page_indexes))
        if not selected:
            return []
        for index in selected:
            if document.page_count is None or index >= document.page_count:
                raise ValueError(f'PDF page index out of range: {index}')
        pdf_bytes = await asyncio.to_thread(document.pdf_path.read_bytes)
        async with self._semaphore:
            response = await asyncio.to_thread(
                ocr_pdf,
                pdf_bytes,
                self._ocr_settings,
                pages=selected,
            )
        try:
            rendered = await asyncio.to_thread(
                _render_pdf_pages,
                document.pdf_path,
                selected,
                self._ocr_settings.render_scale,
                self._max_image_pixels,
                self._max_download_bytes,
            )
        except pdfium.PdfiumError as error:
            raise ExternalRetrievalError(
                'Unable to render external PDF page'
            ) from error
        artifact_pages = await asyncio.to_thread(
            materialize_ocr_artifacts,
            response,
            document.pdf_path,
            self._temporary_directory / 'source-artifacts' / uuid4().hex,
            self._ocr_settings.render_scale,
            self._ocr_settings.block_crop_scale,
            rendered_pages=rendered,
        )
        artifacts_by_index = {page.page_index: page for page in artifact_pages}
        entries: list[ResearchEvidenceEntry] = []
        for page_index in selected:
            page = artifacts_by_index[page_index]
            png_bytes = rendered[page_index]
            spans = markdown_spans(
                page.markdown,
                self._research_budget.counter,
                passage_token_budget=self._passage_token_budget,
            )
            page_parts = [span.raw_content for span in spans] or ['']
            line_offsets = [(span.line_start, span.line_end) for span in spans]
            heading_paths = [list(span.heading_path) for span in spans]
            token_counts = [span.token_count for span in spans]
            if not spans:
                line_offsets = [(None, None)]
                heading_paths = [[]]
                token_counts = [0]
            for (
                raw_content,
                (line_start, line_end),
                heading_path,
                token_count,
            ) in zip(
                page_parts,
                line_offsets,
                heading_paths,
                token_counts,
                strict=True,
            ):
                entries.append(
                    ResearchEvidenceEntry(
                        document_position=0,
                        kind='pdf_page',
                        url=document.url,
                        resolved_url=document.resolved_url,
                        title=document.title,
                        retrieved_at=document.retrieved_at,
                        page_index=page_index,
                        line_start=line_start,
                        line_end=line_end,
                        raw_content=raw_content,
                        heading_path=heading_path,
                        preview=raw_content[:160],
                        content='' if not spans else None,
                        token_count=token_count,
                        media_type='image/png',
                        media_bytes=png_bytes,
                        ocr_artifacts=artifacts_by_index[page_index],
                        rendered_page_bytes=png_bytes,
                    )
                )
        return await self.read_evidence(entries)

    async def read_evidence(
        self,
        entries: list[ResearchEvidenceEntry],
    ) -> list[ResearchEvidenceEntry]:
        """Format complete passages and decode explicitly selected images."""
        results = list(entries)
        text_indexes = [
            index
            for index, entry in enumerate(results)
            if entry.kind != 'image'
            and entry.raw_content is not None
            and entry.content is None
        ]
        if text_indexes:
            raw_values = [
                results[index].raw_content or '' for index in text_indexes
            ]
            raw_costs = self._research_budget.counter.count_texts(raw_values)
            if any(
                cost > self._research_budget.token_limit for cost in raw_costs
            ):
                raise InputBudgetExceeded(
                    'A complete research passage exceeds the input token budget'
                )
            formatted = await self._format_passages(
                [
                    FormattingInput(
                        position=index,
                        content=results[index].raw_content or '',
                    )
                    for index in text_indexes
                ]
            )
            formatted_costs = self._research_budget.counter.count_texts(
                list(formatted.values())
            )
            if any(
                cost > self._research_budget.token_limit
                for cost in formatted_costs
            ):
                raise InputBudgetExceeded(
                    'A complete formatted passage exceeds the input token budget'
                )
            for index, content in formatted.items():
                results[index] = results[index].model_copy(
                    update={'content': content}
                )
        for index, entry in enumerate(results):
            if entry.kind != 'image' or entry.media_bytes is not None:
                continue
            async with self._semaphore:
                downloaded = await self._media_client.download(entry.url)
            (
                png,
                mime_type,
                original,
                original_type,
                visual_note,
            ) = await asyncio.to_thread(
                decode_image,
                downloaded,
                max_pixels=self._max_image_pixels,
                max_bytes=self._max_download_bytes,
            )
            results[index] = entry.model_copy(
                update={
                    'resolved_url': downloaded.final_url,
                    'retrieved_at': downloaded.retrieved_at,
                    'content': visual_note,
                    'media_type': mime_type,
                    'media_bytes': png,
                    'original_media_type': original_type,
                    'original_media_bytes': original,
                }
            )
        return results

    async def _format_passages(
        self,
        requests: list[FormattingInput],
    ) -> dict[int, str]:
        """Run complete raw passages through formatter-sized batches."""
        item_payloads = [
            json.dumps(
                {
                    'position': self._formatting_batch_size - 1,
                    'content': item.content,
                },
                ensure_ascii=False,
                separators=(',', ':'),
            )
            for item in requests
        ]
        counts = self._formatting_budget.counter.count_texts(item_payloads)
        list_overhead = self._formatting_budget.counter.count_texts(['[,]'])[0]
        batches = pack_items(
            requests,
            token_counts=[count + list_overhead for count in counts],
            token_budget=self._formatting_budget.token_limit,
            max_items=self._formatting_batch_size,
        )
        outputs_by_position: dict[int, str] = {}
        for batch in batches:
            indexed_batch = [
                FormattingInput(position=index, content=request.content)
                for index, request in enumerate(batch)
            ]
            async with self._semaphore:
                outputs = await self._formatter.aforward(blocks=indexed_batch)
            outputs_by_position.update(
                (request.position, output.formatted_content)
                for request, output in zip(batch, outputs, strict=True)
            )
        return outputs_by_position


def _pdf_metadata(path: Path) -> tuple[int, str | None]:
    """Read page count/title while owning PDFium for the entire lifetime."""
    with PDFIUM_LOCK:
        pdf = pdfium.PdfDocument(str(path))
        try:
            metadata = pdf.get_metadata_dict()
            title = metadata.get('Title') if metadata else None
            return len(pdf), title or None
        finally:
            pdf.close()


def _search_pdf(path: Path, query: str, limit: int) -> PdfSearchResult:
    """Search page-native text and retain exact source snippets."""
    needle = query.casefold()
    matches: list[PdfSearchMatch] = []
    matching_pages = 0
    with PDFIUM_LOCK:
        pdf = pdfium.PdfDocument(str(path))
        try:
            for page_index in range(len(pdf)):
                page = pdf[page_index]
                try:
                    text_page = page.get_textpage()
                    try:
                        text = text_page.get_text_range()
                    finally:
                        text_page.close()
                finally:
                    page.close()
                folded_parts: list[str] = []
                source_indexes: list[int] = []
                for source_index, character in enumerate(text):
                    folded_character = character.casefold()
                    folded_parts.append(folded_character)
                    source_indexes.extend(
                        [source_index] * len(folded_character)
                    )
                folded = ''.join(folded_parts)
                offset = folded.find(needle)
                if offset < 0:
                    continue
                if needle:
                    start_index = source_indexes[offset]
                    end_index = source_indexes[offset + len(needle) - 1] + 1
                else:
                    start_index = 0
                    end_index = min(1, len(text))
                matching_pages += 1
                if len(matches) < limit:
                    start = max(0, start_index - 100)
                    end = min(len(text), end_index + 100)
                    matches.append(
                        PdfSearchMatch(
                            page_index=page_index,
                            snippet=text[start:end],
                        )
                    )
            return PdfSearchResult(
                matches=matches,
                truncated=matching_pages > limit,
            )
        finally:
            pdf.close()


def _render_pdf_pages(
    path: Path,
    page_indexes: list[int],
    scale: float,
    max_pixels: int,
    max_bytes: int,
) -> dict[int, bytes]:
    """Render selected pages while closing all PDFium and Pillow resources."""
    rendered: dict[int, bytes] = {}
    with PDFIUM_LOCK:
        pdf = pdfium.PdfDocument(str(path))
        try:
            for page_index in page_indexes:
                page = pdf[page_index]
                try:
                    width, height = page.get_size()
                    pixel_width = round(width * scale)
                    pixel_height = round(height * scale)
                    if pixel_width * pixel_height > max_pixels:
                        raise ExternalRetrievalError(
                            f'PDF page {page_index} exceeds the configured pixel limit'
                        )
                    bitmap = page.render(scale=scale)
                    try:
                        image = bitmap.to_pil()
                        try:
                            if image.width * image.height > max_pixels:
                                raise ExternalRetrievalError(
                                    f'PDF page {page_index} exceeds the configured pixel limit'
                                )
                            buffer = io.BytesIO()
                            image.save(buffer, format='PNG')
                            png_bytes = buffer.getvalue()
                            if len(png_bytes) > max_bytes:
                                raise ExternalRetrievalError(
                                    f'PDF page {page_index} exceeds the configured byte limit'
                                )
                            rendered[page_index] = png_bytes
                        finally:
                            image.close()
                    finally:
                        bitmap.close()
                finally:
                    page.close()
        finally:
            pdf.close()
    return rendered
