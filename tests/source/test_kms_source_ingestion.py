"""Behavioral tests for complete capture ingestion and durable provenance."""

import asyncio
import hashlib
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from kms.core.tokenization.budget import InputBudgetExceeded
from kms.source.ingestion import SourceIngestionService
from kms.source.models.inputs import CapturedPDFSource, PDFSourceInput
from kms.source.models.snapshot import SourceSnapshot
from kms.source.models.source import BlockType, SourceBlock, SourcePage
from kms.source.processing.state import SourceProcessingState
from kms.source.storage.artifacts import SourceArtifactStore

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


class _Repository:
    def __init__(self) -> None:
        self.snapshots: dict[str, SourceSnapshot] = {}
        self.created: list[tuple[str, SourceSnapshot]] = []

    async def load_owned_snapshot(self, user_uuid: str, source_uuid: str):
        return self.snapshots.get(source_uuid)

    async def create_owned_snapshot(
        self, user_uuid: str, snapshot: SourceSnapshot
    ):
        self.snapshots[snapshot.source.uuid] = snapshot
        self.created.append((user_uuid, snapshot))
        return snapshot


class _ProcessingGraph:
    def __init__(self, pages=None, error=None) -> None:
        self.pages = pages or []
        self.error = error
        self.calls = []

    async def run(self, state: SourceProcessingState):
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
        return SimpleNamespace(
            source=source,
            split_pages=self.pages,
            instructions=[],
            statements=[],
            procedures=[],
        )


def _service(root, repository=None, processing_graph=None):
    return SourceIngestionService(
        repository or _Repository(),
        SourceArtifactStore(root),
        processing_graph or _ProcessingGraph(),
        embedding_model='local/test',
        embedding_dimension=2,
    )


def _capture(
    *,
    url='https://example.test/source.pdf',
    data=b'%PDF-1.7\nsource',
    pages=None,
):
    return CapturedPDFSource(
        url=url,
        title='source.pdf',
        retrieved_at=_NOW,
        capture_id='capture-123',
        data=data,
    )


def test_exact_page_selector_and_owner_define_capture_identity(tmp_path):
    async def exercise():
        repository = _Repository()
        graph = _ProcessingGraph()
        service = _service(tmp_path / 'store', repository, graph)
        capture = _capture()
        selections = (None, [], [2, 0], [0, 2])
        snapshots = [
            await service.ingest('owner', capture, pages=pages)
            for pages in selections
        ]
        assert len({snapshot.source.uuid for snapshot in snapshots}) == 4
        repeated = await service.ingest('owner', capture, pages=[2, 0])
        assert repeated.source.uuid == snapshots[2].source.uuid
        assert len(graph.calls) == 4
        other = await service.ingest('other-owner', capture, pages=[2, 0])
        assert other.source.uuid != snapshots[2].source.uuid

    asyncio.run(exercise())


def test_ingestion_preserves_full_original_pdf_and_page_selection(tmp_path):
    data = b'%PDF-1.7\ncomplete source'
    page = SourcePage(
        index=2,
        markdown='Source text',
        blocks=[
            SourceBlock(
                block_type=BlockType.PARAGRAPH,
                content='Source text',
                embedding=[0.25, 0.75],
            )
        ],
    )
    repository = _Repository()
    graph = _ProcessingGraph([page])
    service = _service(tmp_path / 'store', repository, graph)
    capture = _capture(data=data)

    result = asyncio.run(service.ingest('owner', capture, pages=[2, 0]))

    assert isinstance(graph.calls[0].input, PDFSourceInput)
    assert graph.calls[0].input.pages == [2, 0]
    assert result.source.metadata['identity_schema'] == 'source-content-v3'
    assert (
        result.source.metadata['captures']['original_pdf_sha256']
        == hashlib.sha256(data).hexdigest()
    )
    assert (
        Path(
            result.source.metadata['captures']['original_pdf_path']
        ).read_bytes()
        == data
    )
    assert result.source.metadata['page_indexes'] == [2]
    assert repository.created == [('owner', result)]
    assert not Path(graph.calls[0].input.pdf_path).exists()


def test_local_capture_has_no_fake_url_and_records_local_origin(tmp_path):
    page = SourcePage(index=0, markdown='', blocks=[])
    service = _service(
        tmp_path / 'store', processing_graph=_ProcessingGraph([page])
    )
    capture = _capture(url=None)
    result = asyncio.run(service.ingest('owner', capture))
    assert result.source.metadata['origin'] == 'local'
    assert result.source.metadata['url'] is None
    assert result.source.key == 'source.pdf'


def test_processing_failure_cleans_temporary_pdf_and_does_not_persist(tmp_path):
    repository = _Repository()
    service = _service(
        tmp_path / 'store',
        repository,
        _ProcessingGraph(
            error=InputBudgetExceeded('candidate processing failed')
        ),
    )
    with pytest.raises(
        InputBudgetExceeded, match='candidate processing failed'
    ):
        asyncio.run(service.ingest('owner', _capture()))
    assert repository.created == []
    assert not service._artifact_store._root.exists()


def test_ingestion_reuses_existing_owned_snapshot_without_processing(tmp_path):
    async def scenario():
        repository = _Repository()
        graph = _ProcessingGraph()
        service = _service(tmp_path / 'store', repository, graph)
        capture = _capture()
        initial = await service.ingest('owner', capture)
        repository.created.clear()
        graph.calls.clear()
        result = await service.ingest('owner', capture)
        assert result == initial
        assert graph.calls == []
        assert repository.created == []

    asyncio.run(scenario())


def test_ingestion_publishes_complete_v3_snapshot(tmp_path):
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
    repository = _Repository()
    graph = _ProcessingGraph([page])
    service = _service(tmp_path / 'store', repository, graph)
    persisted = asyncio.run(
        service.ingest('owner', _capture(data=b'%PDF complete'))
    )
    assert len(graph.calls) == 1
    assert graph.calls[0].input.pages is None
    assert repository.created == [('owner', persisted)]
    assert persisted.pages[0].blocks[0].content == 'Complete source material.'
    assert persisted.pages[0].blocks[0].embedding == [0.25, 0.75]
    assert persisted.source.metadata['processed_manifest_path']
    assert persisted.source.metadata['identity_schema'] == 'source-content-v3'


def test_cancellation_during_processing_cleans_only_temporary_pdf(tmp_path):
    class _BlockingGraph:
        def __init__(self):
            self.started = asyncio.Event()
            self.pdf_path = None

        async def run(self, state):
            self.pdf_path = Path(state.input.pdf_path)
            self.started.set()
            await asyncio.Future()

    async def scenario():
        graph = _BlockingGraph()
        repository = _Repository()
        service = _service(tmp_path / 'store', repository, graph)
        task = asyncio.create_task(service.ingest('owner', _capture()))
        await graph.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not graph.pdf_path.exists()
        assert repository.created == []
        assert not (tmp_path / 'store').exists()

    asyncio.run(scenario())
