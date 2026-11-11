"""Process captured PDFs and atomically publish complete owned snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from functools import partial
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from kms.concurrency import run_blocking_work
from kms.source.models.inputs import CapturedPDFSource, PDFSourceInput
from kms.source.models.snapshot import (
    SourceSnapshot,
    assert_source_embedding_profile,
    finalize_snapshot_identities,
)
from kms.source.models.source import Source
from kms.source.processing.graph import SourceProcessingGraph
from kms.source.processing.state import SourceProcessingState
from kms.source.storage.artifacts import SourceArtifactStore
from kms.source.storage.snapshots import (
    SourceSnapshotRepository,
)


class SourceIngestionService:
    """Own the processing and transactional publication of complete snapshots."""

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

    async def ingest(
        self,
        user_uuid: str,
        capture: CapturedPDFSource,
        *,
        pages: list[int] | None = None,
    ) -> SourceSnapshot:
        """Reuse or process a capture, then publish one complete source snapshot."""
        descriptor, temporary_name = tempfile.mkstemp(suffix='.pdf')
        os.close(descriptor)
        pdf_path = Path(temporary_name)
        try:
            await run_blocking_work(pdf_path.write_bytes, capture.data)
        except BaseException:
            pdf_path.unlink(missing_ok=True)
            raise
        try:
            capture_digest = _capture_digest(capture, pages)
            source = Source(
                uuid=_source_uuid(user_uuid, capture_digest),
                key=capture.title or capture.url,
                metadata={
                    'origin': 'local' if capture.url is None else 'research',
                    'url': capture.url,
                    'resolved_url': capture.resolved_url,
                    'retrieved_at': capture.retrieved_at.isoformat(),
                    'capture_id': capture.capture_id,
                    'snapshot_digest': capture_digest,
                },
            )
            existing = await self._repository.load_owned_snapshot(
                user_uuid, source.uuid
            )
            if existing is not None:
                assert_source_embedding_profile(
                    existing,
                    model=self._embedding_model,
                    dimension=self._embedding_dimension,
                )
                return existing
            processed = await self._processing_graph.run(
                SourceProcessingState(
                    input=PDFSourceInput(pdf_path=str(pdf_path), pages=pages),
                    source=source,
                )
            )
            durable_pages = await run_blocking_work(
                self._artifact_store.finalize_pages, processed.split_pages
            )
            snapshot = finalize_snapshot_identities(
                processed.source,
                durable_pages,
                processed.instructions,
                processed.statements,
                processed.procedures,
            )
            assert_source_embedding_profile(
                snapshot,
                model=self._embedding_model,
                dimension=self._embedding_dimension,
            )
            pdf_artifact = await run_blocking_work(
                partial(self._artifact_store.write, capture.data, suffix='.pdf')
            )
            source_metadata = {
                **snapshot.source.metadata,
                'captures': {
                    'original_pdf_path': str(pdf_artifact),
                    'original_pdf_sha256': hashlib.sha256(
                        capture.data
                    ).hexdigest(),
                },
                'page_count': len(snapshot.pages),
                'page_indexes': [page.index for page in snapshot.pages],
            }
            snapshot = snapshot.model_copy(
                update={
                    'source': snapshot.source.model_copy(
                        update={'metadata': source_metadata}
                    )
                }
            )
            manifest = json.dumps(
                snapshot.model_dump(mode='json'),
                sort_keys=True,
                ensure_ascii=False,
                separators=(',', ':'),
                allow_nan=False,
            ).encode('utf-8')
            manifest_path = await run_blocking_work(
                partial(self._artifact_store.write, manifest, suffix='.json')
            )
            snapshot = snapshot.model_copy(
                update={
                    'source': snapshot.source.model_copy(
                        update={
                            'metadata': {
                                **snapshot.source.metadata,
                                'processed_manifest_path': str(manifest_path),
                            }
                        }
                    )
                }
            )
            return await self._repository.create_owned_snapshot(
                user_uuid, snapshot
            )
        finally:
            pdf_path.unlink()


def _source_uuid(user_uuid: str, capture_digest: str) -> str:
    return str(
        uuid5(
            NAMESPACE_URL, f'kms-source-snapshot:{user_uuid}:{capture_digest}'
        )
    )


def _capture_digest(capture: CapturedPDFSource, pages: list[int] | None) -> str:
    canonical = json.dumps(
        {
            'schema': 'source-capture-v3',
            'pdf_sha256': hashlib.sha256(capture.data).hexdigest(),
            'url': capture.url,
            'pages': pages,
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(',', ':'),
        allow_nan=False,
    ).encode('utf-8')
    return hashlib.sha256(canonical).hexdigest()
