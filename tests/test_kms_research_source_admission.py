"""Behavioral tests for complete capture admission and durable provenance."""

import asyncio
import hashlib
import threading
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from kms.core.model.block import SourceBlock
from kms.core.model.block_types import BlockType
from kms.core.model.page import SourcePage
from kms.core.model.source import Source
from kms.core.model.source_processing.admission import (
    CapturedPDFSource,
    SourceSnapshot,
    SourceSnapshotPreparation,
)
from kms.core.model.source_processing.input import PDFSourceInput
from kms.core.tokenization.budget import InputBudgetExceeded
from kms.langgraph.source_processing.state import SourceProcessingState
from kms.service.source_admission import SourceAdmissionService
from kms.service.source_artifacts import SourceArtifactStore

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


class _Repository:
    def __init__(self) -> None:
        self.snapshots: dict[str, SourceSnapshot] = {}
        self.created: list[tuple[str, SourceSnapshot]] = []

    async def load_owned_snapshot(
        self, user_uuid: str, source_uuid: str
    ) -> SourceSnapshot | None:
        return self.snapshots.get(source_uuid)

    async def create_owned_snapshot(
        self, user_uuid: str, source: Source, pages: list[SourcePage]
    ) -> SourceSnapshot:
        snapshot = SourceSnapshot(source=source, pages=pages)
        self.snapshots[source.uuid] = snapshot
        self.created.append((user_uuid, snapshot))
        return snapshot


class _ProcessingGraph:
    def __init__(self, pages=None, error=None) -> None:
        self.pages = pages or []
        self.error = error
        self.calls = []

    async def run(self, state: SourceProcessingState) -> SourceProcessingState:
        self.calls.append(state)
        if self.error is not None:
            raise self.error
        source = state.source.model_copy(
            update={
                'metadata': {
                    **state.source.metadata,
                    'embedding_profile': {
                        'model': 'local/test',
                        'dimension': 2,
                    },
                }
            }
        )
        return state.model_copy(
            update={'source': source, 'embedded_pages': self.pages}
        )


def _service(
    root: Path,
    repository: _Repository | None = None,
    processing_graph: _ProcessingGraph | None = None,
) -> SourceAdmissionService:
    return SourceAdmissionService(
        repository or _Repository(),
        SourceArtifactStore(root),
        processing_graph or _ProcessingGraph(),
        embedding_model='local/test',
        embedding_dimension=2,
    )


def test_pdf_capture_uses_full_original_file_and_records_hash(tmp_path):
    data = b'%PDF-1.7\ncomplete source'
    capture = CapturedPDFSource(
        url='https://example.test/book.pdf',
        resolved_url='https://cdn.example.test/book.pdf',
        title='Book',
        retrieved_at=_NOW,
        capture_id='capture-123',
        data=data,
    )

    store_root = tmp_path / 'store'
    service = _service(store_root)
    preparation = asyncio.run(service.prepare('user', capture))

    assert isinstance(preparation.input, PDFSourceInput)
    assert preparation.input.pages is None
    assert Path(preparation.input.pdf_path).read_bytes() == data
    assert not store_root.exists()
    service.cleanup(capture.capture_id)


def test_failed_processing_cleans_temporary_pdf_and_does_not_persist(
    tmp_path,
):
    store_root = tmp_path / 'store'
    service = _service(
        store_root,
        processing_graph=_ProcessingGraph(
            error=InputBudgetExceeded('candidate processing failed')
        ),
    )
    capture = CapturedPDFSource(
        url='https://example.test/source.pdf',
        retrieved_at=_NOW,
        data=b'%PDF-1.7\ncandidate',
    )

    with pytest.raises(
        InputBudgetExceeded, match='candidate processing failed'
    ):
        asyncio.run(service.admit('user', capture))

    assert not store_root.exists()
    assert not service._temporary_pdf_paths


def test_failed_snapshot_lookup_cleans_prepared_pdf(tmp_path):
    class _FailingRepository(_Repository):
        async def load_owned_snapshot(self, user_uuid, source_uuid):
            raise RuntimeError('snapshot lookup failed')

    service = _service(tmp_path / 'store', _FailingRepository())
    capture = CapturedPDFSource(
        url='https://example.test/source.pdf',
        retrieved_at=_NOW,
        data=b'%PDF-1.7\ncandidate',
    )

    with pytest.raises(RuntimeError, match='snapshot lookup failed'):
        asyncio.run(service.admit('user', capture))

    assert service._temporary_pdf_paths == {}


def test_cancellation_during_prepare_waits_for_and_cleans_pdf(
    tmp_path, monkeypatch
):
    entered = threading.Event()
    release = threading.Event()

    async def scenario():
        service = _service(tmp_path / 'store')
        original_prepare_pdf = service._prepare_pdf

        def blocking_prepare_pdf(user_uuid, capture):
            entered.set()
            release.wait()
            return original_prepare_pdf(user_uuid, capture)

        monkeypatch.setattr(service, '_prepare_pdf', blocking_prepare_pdf)
        capture = CapturedPDFSource(
            url='https://example.test/source.pdf',
            retrieved_at=_NOW,
            data=b'%PDF-1.7\ncandidate',
        )
        task = asyncio.create_task(service.admit('user', capture))
        await asyncio.to_thread(entered.wait)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert service._temporary_pdf_paths == {}

    asyncio.run(scenario())


def test_admit_reuses_existing_owned_snapshot_without_processing(tmp_path):
    async def scenario():
        repository = _Repository()
        graph = _ProcessingGraph()
        service = _service(tmp_path / 'store', repository, graph)
        capture = CapturedPDFSource(
            url='https://example.test/source.pdf',
            retrieved_at=_NOW,
            data=b'%PDF-1.7\ncandidate',
        )
        preparation = await service.prepare('user', capture)
        assert isinstance(preparation, SourceSnapshotPreparation)
        service.cleanup(capture.capture_id)
        snapshot = SourceSnapshot(
            source=preparation.source.model_copy(
                update={
                    'metadata': {
                        **preparation.source.metadata,
                        'embedding_profile': {
                            'model': 'local/test',
                            'dimension': 2,
                        },
                    }
                }
            ),
            pages=[],
        )
        repository.snapshots[snapshot.source.uuid] = snapshot

        result = await service.admit('user', capture)

        assert result == snapshot
        assert graph.calls == []
        assert service._temporary_pdf_paths == {}

    asyncio.run(scenario())


def test_admit_processes_and_finalizes_complete_pages(tmp_path):
    page = SourcePage(
        index=3,
        markdown='Complete source material.',
        blocks=[
            SourceBlock(
                block_type=BlockType.PARAGRAPH,
                content='Complete source material.',
                embedding=[0.25, 0.75],
            )
        ],
    )

    async def scenario():
        repository = _Repository()
        graph = _ProcessingGraph([page])
        service = _service(tmp_path / 'store', repository, graph)
        capture = CapturedPDFSource(
            url='https://example.test/source.pdf',
            retrieved_at=_NOW,
            data=b'%PDF-1.7\ncomplete source',
        )

        persisted = await service.admit('owner', capture)

        assert len(graph.calls) == 1
        assert graph.calls[0].input.pages is None
        assert repository.created == [('owner', persisted)]
        assert persisted.pages[0].index == 3
        assert (
            persisted.pages[0].blocks[0].content == 'Complete source material.'
        )
        assert persisted.pages[0].blocks[0].embedding == [0.25, 0.75]
        assert persisted.source.metadata['processed_manifest_path']
        assert service._temporary_pdf_paths == {}

    asyncio.run(scenario())


def test_admit_cleans_temporary_pdf_on_cancellation(tmp_path):
    entered = asyncio.Event()

    class _BlockingGraph:
        async def run(self, state):
            entered.set()
            await asyncio.Event().wait()

    async def scenario():
        service = _service(
            tmp_path / 'store', processing_graph=_BlockingGraph()
        )
        capture = CapturedPDFSource(
            url='https://example.test/source.pdf',
            retrieved_at=_NOW,
            data=b'%PDF-1.7\ncandidate',
        )
        task = asyncio.create_task(service.admit('user', capture))
        await entered.wait()
        temporary_paths = tuple(service._temporary_pdf_paths.values())
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert service._temporary_pdf_paths == {}
        assert all(not path.exists() for path in temporary_paths)

    asyncio.run(scenario())


def test_finalize_persists_complete_snapshot_before_return(tmp_path):
    repository = _Repository()
    service = _service(tmp_path / 'store', repository)
    source = Source(
        uuid=str(uuid4()),
        key='complete source',
        metadata={'embedding_profile': {'model': 'local/test', 'dimension': 2}},
    )
    capture = CapturedPDFSource(
        url='https://example.test/source.pdf',
        retrieved_at=_NOW,
        data=b'%PDF-1.7\ncomplete source',
    )
    page = SourcePage(
        index=3,
        markdown='Complete source material.',
        blocks=[
            SourceBlock(
                block_type=BlockType.PARAGRAPH,
                content='Complete source material.',
                embedding=[0.25, 0.75],
            )
        ],
    )
    preparation = SourceSnapshotPreparation(
        source=source,
        input=PDFSourceInput(pdf_path='capture.pdf'),
    )
    persisted = asyncio.run(
        service.finalize('owner', capture, preparation, [page])
    )

    assert repository.created == [('owner', persisted)]
    assert repository.snapshots[source.uuid] == persisted
    assert persisted.pages[0].blocks[0].uuid
    assert persisted.source.metadata['processed_manifest_path']
    capture_record = persisted.source.metadata['captures']
    assert (
        Path(capture_record['original_pdf_path']).read_bytes() == capture.data
    )
    assert (
        capture_record['original_pdf_sha256']
        == hashlib.sha256(capture.data).hexdigest()
    )
