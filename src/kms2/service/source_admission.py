"""Prepare research captures as durable, unpublished source snapshots."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
from pathlib import Path
from urllib.parse import urldefrag, urljoin
from uuid import NAMESPACE_URL, uuid4, uuid5

from markdown_it import MarkdownIt

from kms2.core.model.block import SourceBlock
from kms2.core.model.block_types import BlockType
from kms2.core.model.page import SourcePage
from kms2.core.model.source import Source
from kms2.core.model.source_processing.admission import (
    ResearchPDFMaterial,
    ResearchSourceMaterial,
    ResearchWebMaterial,
    SourceAdmissionResult,
    SourceSnapshotPreparation,
    StagedSourceSnapshot,
)
from kms2.core.model.source_processing.input import (
    OCRSourceInput,
    TextSourceInput,
)
from kms2.core.model.visual_asset import VisualAsset
from kms2.core.research_text import markdown_spans
from kms2.core.source_snapshot import finalize_snapshot_identities
from kms2.core.windowing import TextTokenCounter
from kms2.database.source.source_snapshot_repository import (
    SourceSnapshotRepository,
)


class SourceAdmissionService:
    """Capture and prepare durable research snapshots without publishing them."""

    def __init__(
        self,
        repository: SourceSnapshotRepository,
        token_counter: TextTokenCounter,
        storage_root: str | Path,
        passage_token_budget: int,
    ) -> None:
        self._repository = repository
        self._token_counter = token_counter
        self._storage_root = Path(storage_root)
        self._passage_token_budget = passage_token_budget

    async def prepare(
        self,
        user_uuid: str,
        material: ResearchSourceMaterial,
    ) -> SourceAdmissionResult | SourceSnapshotPreparation:
        """Durably capture one material and reuse an exact owned snapshot."""
        preparation = await asyncio.to_thread(
            self._prepare_files, user_uuid, material
        )
        existing = await self._repository.load_owned_snapshot(
            user_uuid, preparation.source.uuid
        )
        return existing if existing is not None else preparation

    async def finalize(
        self,
        preparation: SourceSnapshotPreparation,
        pages: list[SourcePage],
    ) -> StagedSourceSnapshot:
        """Durably store processed media and assign final child identities."""
        durable_pages = await asyncio.to_thread(self._durable_pages, pages)
        return finalize_snapshot_identities(preparation.source, durable_pages)

    def _durable_pages(self, pages: list[SourcePage]) -> list[SourcePage]:
        durable_pages = []
        for page in pages:
            blocks = []
            for block in page.blocks:
                crop_path = (
                    str(
                        self._store(
                            Path(block.crop_path).read_bytes(),
                            suffix=Path(block.crop_path).suffix or '.bin',
                        )
                    )
                    if block.crop_path is not None
                    else None
                )
                assets = [
                    asset.model_copy(
                        update={
                            'path': str(
                                self._store(
                                    Path(asset.path).read_bytes(),
                                    suffix=Path(asset.path).suffix or '.bin',
                                )
                            )
                        }
                    )
                    for asset in block.assets
                ]
                blocks.append(
                    block.model_copy(
                        update={'crop_path': crop_path, 'assets': assets}
                    )
                )
            durable_pages.append(page.model_copy(update={'blocks': blocks}))
        return durable_pages

    def _prepare_files(
        self,
        user_uuid: str,
        material: ResearchSourceMaterial,
    ) -> SourceSnapshotPreparation:
        if isinstance(material, ResearchWebMaterial):
            return self._prepare_web(user_uuid, material)
        return self._prepare_pdf(user_uuid, material)

    def _prepare_web(
        self, user_uuid: str, material: ResearchWebMaterial
    ) -> SourceSnapshotPreparation:
        spans = markdown_spans(
            material.raw_markdown,
            self._token_counter,
            passage_token_budget=self._passage_token_budget,
        )
        image_by_url = {}
        image_hashes = []
        for image in material.images:
            resolved = urldefrag(urljoin(material.url, image.url))[0]
            if resolved in image_by_url:
                continue
            png_path = self._store(image.media_bytes, suffix='.png')
            original_path = (
                self._store(image.original_media_bytes)
                if image.original_media_bytes is not None
                else None
            )
            image_by_url[resolved] = (image, png_path, original_path)
            original_hash = (
                hashlib.sha256(image.original_media_bytes).hexdigest()
                if image.original_media_bytes is not None
                else ''
            )
            image_resolved = (
                urldefrag(image.resolved_url)[0]
                if image.resolved_url is not None
                else resolved
            )
            image_hashes.append(
                hashlib.sha256(
                    (
                        f'{resolved}:{image_resolved}:'
                        f'{hashlib.sha256(image.media_bytes).hexdigest()}:'
                        f'{original_hash}'
                    ).encode()
                ).hexdigest()
            )

        parser = MarkdownIt('commonmark').enable('table')
        image_lines: dict[str, int] = {}
        image_order: dict[str, int] = {}
        for token in parser.parse(material.raw_markdown):
            if token.type != 'inline' or token.children is None:
                continue
            for child in token.children:
                reference = (
                    child.attrGet('src') if child.type == 'image' else None
                )
                if reference:
                    absolute = urldefrag(urljoin(material.url, reference))[0]
                    image_lines.setdefault(
                        absolute, token.map[0] + 1 if token.map else 1
                    )
                    image_order.setdefault(absolute, len(image_order))

        linked_image_urls = sorted(
            image_by_url,
            key=lambda image_url: (
                image_lines.get(image_url, 1 << 60),
                image_order.get(image_url, 1 << 60),
                image_url,
            ),
        )
        ordered_blocks: list[SourceBlock] = []
        placed: set[str] = set()
        for span in spans:
            ordered_blocks.append(
                SourceBlock(
                    block_type=BlockType.MARKDOWN, content=span.raw_content
                )
            )
            for image_url in linked_image_urls:
                _, path, _ = image_by_url[image_url]
                line = image_lines.get(image_url)
                if (
                    image_url not in placed
                    and line is not None
                    and span.line_start <= line <= span.line_end
                ):
                    ordered_blocks.append(
                        SourceBlock(
                            block_type=BlockType.IMAGE,
                            content=None,
                            assets=[VisualAsset(path=str(path))],
                        )
                    )
                    placed.add(image_url)
        for image_url in sorted(image_by_url):
            _, path, _ = image_by_url[image_url]
            if image_url not in placed:
                ordered_blocks.append(
                    SourceBlock(
                        block_type=BlockType.IMAGE,
                        content=None,
                        assets=[VisualAsset(path=str(path))],
                    )
                )

        markdown_path = self._store(
            material.raw_markdown.encode(), suffix='.md'
        )
        capture_hashes = sorted(image_hashes)
        digest = self._digest(
            material.url,
            material.resolved_url,
            'web',
            material.raw_markdown,
            capture_hashes,
        )
        captures = {
            'images': [
                {
                    'url': url,
                    'resolved_url': image.resolved_url,
                    'png_path': str(path),
                    'original_path': str(original_path)
                    if original_path
                    else None,
                    'media_type': 'image/png',
                    'original_media_type': image.original_media_type,
                }
                for url, (image, path, original_path) in sorted(
                    image_by_url.items()
                )
            ]
        }
        source = Source(
            uuid=str(
                uuid5(
                    NAMESPACE_URL, f'kms2-source-snapshot:{user_uuid}:{digest}'
                )
            ),
            key=material.title or material.url,
            metadata={
                'origin': 'research',
                'url': material.url,
                'resolved_url': material.resolved_url,
                'retrieved_at': material.retrieved_at.isoformat(),
                'snapshot_digest': digest,
                'selected_page_indexes': [],
                'capture_scope': 'web_page',
                'captures': captures,
                'raw_markdown_path': str(markdown_path),
            },
        )
        return SourceSnapshotPreparation(
            source=source,
            input=TextSourceInput(
                pages=[
                    SourcePage(
                        index=0,
                        markdown=material.raw_markdown,
                        blocks=ordered_blocks,
                    )
                ]
            ),
        )

    def _prepare_pdf(
        self, user_uuid: str, material: ResearchPDFMaterial
    ) -> SourceSnapshotPreparation:
        digest_parts: list[str] = []
        selected_indexes = sorted(page.page_index for page in material.pages)
        captures: dict[str, object] = {'pages': []}
        prepared_pages = []
        for page in sorted(material.pages, key=lambda item: item.page_index):
            digest_parts.extend(
                [
                    f'page-index:{page.page_index}',
                    f'page-markdown:{hashlib.sha256(page.markdown.encode()).hexdigest()}',
                    f'page-render:{hashlib.sha256(page.rendered_page).hexdigest()}',
                ]
            )
            page_markdown_path = self._store(
                page.markdown.encode(), suffix='.md'
            )
            rendered_path = self._store(page.rendered_page, suffix='.png')
            blocks = []
            page_image_hashes = []
            for artifact in page.artifacts.blocks:
                crop_path = (
                    self._store(
                        Path(artifact.crop_path).read_bytes(), suffix='.png'
                    )
                    if artifact.crop_path is not None
                    else None
                )
                assets = []
                for image in artifact.images:
                    image_bytes = Path(image.path).read_bytes()
                    page_image_hashes.append(
                        hashlib.sha256(image_bytes).hexdigest()
                    )
                    assets.append(
                        VisualAsset(
                            path=str(self._store(image_bytes, suffix='.png'))
                        )
                    )
                blocks.append(
                    SourceBlock(
                        block_type=artifact.block_type,
                        content=artifact.content,
                        crop_path=str(crop_path) if crop_path else None,
                        crop_bbox=artifact.crop_bbox,
                        assets=assets,
                    )
                )
            prepared_pages.append(
                SourcePage(
                    index=page.page_index, markdown=page.markdown, blocks=blocks
                )
            )
            captures['pages'].append(
                {
                    'page_index': page.page_index,
                    'rendered_path': str(rendered_path),
                    'markdown_path': str(page_markdown_path),
                }
            )
            digest_parts.extend(
                f'ocr-image:{value}' for value in sorted(page_image_hashes)
            )
        digest = self._digest(
            material.url,
            material.resolved_url,
            'pdf',
            json.dumps(
                [
                    selected_indexes,
                    [
                        [page.page_index, page.markdown]
                        for page in sorted(
                            material.pages, key=lambda item: item.page_index
                        )
                    ],
                ],
                ensure_ascii=False,
                separators=(',', ':'),
            ),
            digest_parts,
        )
        source = Source(
            uuid=str(
                uuid5(
                    NAMESPACE_URL, f'kms2-source-snapshot:{user_uuid}:{digest}'
                )
            ),
            key=material.title or material.url,
            metadata={
                'origin': 'research',
                'url': material.url,
                'resolved_url': material.resolved_url,
                'retrieved_at': material.retrieved_at.isoformat(),
                'snapshot_digest': digest,
                'selected_page_indexes': selected_indexes,
                'capture_scope': 'selected_pdf_pages',
                'captures': captures,
            },
        )
        return SourceSnapshotPreparation(
            source=source,
            input=OCRSourceInput(pages=prepared_pages),
        )

    def _store(self, data: bytes, *, suffix: str = '.bin') -> Path:
        """Atomically store bytes at a content-addressed path."""
        digest = hashlib.sha256(data).hexdigest()
        path = self._storage_root / digest[:2] / f'{digest}{suffix}'
        if path.exists():
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent, prefix=f'{digest}.{uuid4().hex}.', suffix='.tmp'
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            temporary.write_bytes(data)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    @staticmethod
    def _digest(
        url: str,
        resolved_url: str | None,
        kind: str,
        raw_content: str,
        content_hashes: list[str],
    ) -> str:
        canonical = json.dumps(
            [
                url,
                resolved_url,
                kind,
                raw_content,
                sorted(content_hashes),
            ],
            ensure_ascii=False,
            separators=(',', ':'),
        )
        return hashlib.sha256(canonical.encode()).hexdigest()
