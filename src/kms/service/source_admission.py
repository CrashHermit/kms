"""Prepare captured material for immediate admission as owned source snapshots."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from kms.core.model.page import SourcePage
from kms.core.model.source import Source
from kms.core.model.source_processing.admission import (
    CapturedPDFSource,
    SourceSnapshot,
    SourceSnapshotPreparation,
)
from kms.core.model.source_processing.input import PDFSourceInput
from kms.core.source_snapshot import (
    assert_source_embedding_profile,
    finalize_snapshot_identities,
)
from kms.database.source.source_snapshot_repository import (
    SourceSnapshotRepository,
)
from kms.langgraph.source_processing.graph import SourceProcessingGraph
from kms.langgraph.source_processing.state import SourceProcessingState
from kms.service.source_artifacts import SourceArtifactStore


class SourceAdmissionService:
    """Prepare, process, and persist selected complete source snapshots."""

    def __init__(
        self,
        repository: SourceSnapshotRepository,
        artifacts: SourceArtifactStore,
        processing_graph: SourceProcessingGraph,
        *,
        embedding_model: str,
        embedding_dimension: int,
    ) -> None:
        self._repository = repository
        self._artifact_store = artifacts
        self._processing_graph = processing_graph
        self._embedding_model = embedding_model
        self._embedding_dimension = embedding_dimension
        self._temporary_pdf_paths: dict[str, Path] = {}

    def cleanup(self, capture_id: str) -> None:
        """Remove temporary PDF input after admission succeeds or fails."""
        path = self._temporary_pdf_paths.pop(capture_id, None)
        if path is not None:
            path.unlink(missing_ok=True)

    async def prepare(
        self,
        user_uuid: str,
        capture: CapturedPDFSource,
    ) -> SourceSnapshot | SourceSnapshotPreparation:
        """Reuse an exact owned snapshot or prepare captured PDF input."""
        task = asyncio.create_task(
            asyncio.to_thread(self._prepare_pdf, user_uuid, capture)
        )
        try:
            preparation = await asyncio.shield(task)
        except asyncio.CancelledError:
            try:
                await task
            except Exception:
                pass
            raise
        existing = await self._repository.load_owned_snapshot(
            user_uuid, preparation.source.uuid
        )
        if existing is not None:
            assert_source_embedding_profile(
                existing,
                model=self._embedding_model,
                dimension=self._embedding_dimension,
            )
            return existing
        return preparation

    async def admit(
        self, user_uuid: str, capture: CapturedPDFSource
    ) -> SourceSnapshot:
        """Reuse an owned snapshot or process and persist this capture."""
        try:
            preparation = await self.prepare(user_uuid, capture)
            if isinstance(preparation, SourceSnapshot):
                return preparation

            processed = await self._processing_graph.run(
                SourceProcessingState(
                    input=preparation.input,
                    source=preparation.source,
                )
            )
            return await self.finalize(
                user_uuid,
                capture,
                preparation.model_copy(update={'source': processed.source}),
                processed.embedded_pages,
            )
        finally:
            self.cleanup(capture.capture_id)

    async def finalize(
        self,
        user_uuid: str,
        capture: CapturedPDFSource,
        preparation: SourceSnapshotPreparation,
        pages: list[SourcePage],
    ) -> SourceSnapshot:
        """Finalize and persist the complete source snapshot."""
        durable_pages = await asyncio.to_thread(
            self._artifact_store.finalize_pages, pages
        )
        candidate = finalize_snapshot_identities(
            preparation.source, durable_pages
        )
        assert_source_embedding_profile(
            candidate,
            model=self._embedding_model,
            dimension=self._embedding_dimension,
        )
        pdf_path = await asyncio.to_thread(
            self._artifact_store.store, capture.data, suffix='.pdf'
        )
        capture_metadata = {
            **candidate.source.metadata,
            'captures': {
                'original_pdf_path': str(pdf_path),
                'original_pdf_sha256': hashlib.sha256(capture.data).hexdigest(),
            },
        }
        candidate = candidate.model_copy(
            update={
                'source': candidate.source.model_copy(
                    update={'metadata': capture_metadata}
                )
            }
        )
        metadata = {
            **candidate.source.metadata,
            'page_count': len(candidate.pages),
            'page_indexes': [page.index for page in candidate.pages],
        }
        candidate = candidate.model_copy(
            update={
                'source': candidate.source.model_copy(
                    update={'metadata': metadata}
                )
            }
        )
        manifest = json.dumps(
            {
                'source_uuid': candidate.source.uuid,
                'source': candidate.source.model_dump(mode='json'),
                'pages': [
                    page.model_dump(mode='json') for page in candidate.pages
                ],
            },
            sort_keys=True,
            ensure_ascii=False,
            separators=(',', ':'),
            allow_nan=False,
        ).encode('utf-8')
        manifest_path = await asyncio.to_thread(
            self._artifact_store.store, manifest, suffix='.json'
        )
        metadata = {
            **candidate.source.metadata,
            'processed_manifest_path': str(manifest_path),
        }
        candidate = candidate.model_copy(
            update={
                'source': candidate.source.model_copy(
                    update={'metadata': metadata}
                )
            }
        )
        return await self._repository.create_owned_snapshot(
            user_uuid, candidate.source, candidate.pages
        )

    def _prepare_pdf(
        self,
        user_uuid: str,
        capture: CapturedPDFSource,
    ) -> SourceSnapshotPreparation:
        """Prepare a captured PDF for standard source processing."""
        capture_digest = _capture_digest(
            'pdf', capture.url, hashlib.sha256(capture.data).hexdigest()
        )
        descriptor, temporary_name = tempfile.mkstemp(suffix='.pdf')
        pdf_path = Path(temporary_name)
        self._temporary_pdf_paths[capture.capture_id] = pdf_path
        with os.fdopen(descriptor, 'wb') as temporary_file:
            temporary_file.write(capture.data)
        source_input = PDFSourceInput(pdf_path=str(pdf_path), pages=None)
        source = Source(
            uuid=_source_uuid(user_uuid, capture_digest),
            key=capture.title or capture.url,
            metadata={
                'origin': 'research',
                'url': capture.url,
                'resolved_url': capture.resolved_url,
                'retrieved_at': capture.retrieved_at.isoformat(),
                'capture_id': capture.capture_id,
                'snapshot_digest': capture_digest,
            },
        )
        return SourceSnapshotPreparation(
            source=source,
            input=source_input,
        )


def _source_uuid(user_uuid: str, capture_digest: str) -> str:
    return str(
        uuid5(
            NAMESPACE_URL,
            f'kms2-source-snapshot:{user_uuid}:{capture_digest}',
        )
    )


def _capture_digest(kind: str, url: str, content_identity: object) -> str:
    canonical = json.dumps(
        {
            'schema': 'source-capture-v2',
            'kind': kind,
            'url': url,
            'content': content_identity,
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(',', ':'),
        allow_nan=False,
    ).encode('utf-8')
    return hashlib.sha256(canonical).hexdigest()
