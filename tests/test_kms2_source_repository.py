import asyncio

import pytest

from kms2.core.model.block import SourceBlock
from kms2.core.model.source import Source
from kms2.core.model.visual_asset import VisualAsset
from kms2.database.source.source_block_repository import SourceBlockRepository
from kms2.database.source.source_catalog_repository import (
    SourceCatalogRepository,
    SourceOwnershipConflict,
)
from kms2.database.source.source_graph_repository import SourceGraphRepository


class _RecordingResult:
    def __init__(self, rows: list[dict[str, object]] | None = None) -> None:
        self._rows = rows or []

    async def consume(self) -> None:
        return None

    async def data(self) -> list[dict[str, object]]:
        return self._rows

    async def single(self, *, strict: bool = False) -> dict[str, object]:
        if strict:
            assert len(self._rows) == 1
        return self._rows[0]


class _RecordingSession:
    def __init__(self, result: _RecordingResult | None = None) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self._result = result or _RecordingResult()

    async def run(self, query: str, **parameters: object) -> _RecordingResult:
        self.calls.append((query, parameters))
        return self._result


class _SessionContext:
    def __init__(self, session: _RecordingSession) -> None:
        self._session = session

    async def __aenter__(self) -> _RecordingSession:
        return self._session

    async def __aexit__(self, *args: object) -> None:
        return None


def test_load_pages_restores_page_block_and_asset_order():
    rows = [
        {
            'pages': [
                {
                    'index': 0,
                    'markdown': 'first page',
                    'blocks': [
                        {
                            'uuid': 'block-1',
                            'block_type': 'paragraph',
                            'content': 'first block',
                            'embedding': [0.1, 0.2],
                            'crop_path': 'crop.png',
                            'crop_bbox': [1.0, 2.0, 3.0, 4.0],
                            'assets': [
                                {'uuid': 'asset-1', 'path': 'first.png'},
                                {'uuid': 'asset-2', 'path': 'second.png'},
                            ],
                        }
                    ],
                },
                {'index': 1, 'markdown': 'empty page', 'blocks': []},
            ]
        }
    ]
    session = _RecordingSession(_RecordingResult(rows))

    pages = asyncio.run(
        SourceGraphRepository(lambda: _SessionContext(session)).load_pages(
            'source-1'
        )
    )

    assert [page.index for page in pages] == [0, 1]
    assert [page.markdown for page in pages] == ['first page', 'empty page']
    assert pages[0].blocks == [
        SourceBlock(
            uuid='block-1',
            block_type='paragraph',
            content='first block',
            embedding=[0.1, 0.2],
            crop_path='crop.png',
            crop_bbox=(1.0, 2.0, 3.0, 4.0),
            assets=[
                VisualAsset(uuid='asset-1', path='first.png'),
                VisualAsset(uuid='asset-2', path='second.png'),
            ],
        )
    ]
    assert pages[1].blocks == []
    assert session.calls[0][1] == {'source_uuid': 'source-1'}


def test_list_sources_returns_stable_user_source_summaries():
    session = _RecordingSession(
        _RecordingResult(
            [
                {'uuid': 'source-2', 'key': 'second.pdf', 'revision': 2},
                {'uuid': 'source-1', 'key': 'first.pdf', 'revision': 0},
            ]
        )
    )

    sources = asyncio.run(
        SourceCatalogRepository(lambda: _SessionContext(session)).list_sources(
            'user-1'
        )
    )

    assert sources == [
        Source(uuid='source-2', key='second.pdf', revision=2),
        Source(uuid='source-1', key='first.pdf', revision=0),
    ]


def test_list_unowned_sources_excludes_owned_source_catalog():
    session = _RecordingSession(
        _RecordingResult(
            [{'uuid': 'legacy-1', 'key': 'legacy.pdf', 'revision': 0}]
        )
    )

    sources = asyncio.run(
        SourceCatalogRepository(
            lambda: _SessionContext(session)
        ).list_unowned_sources()
    )

    assert sources == [Source(uuid='legacy-1', key='legacy.pdf', revision=0)]


def test_adopt_source_locks_source_and_rejects_claimed_source():
    session = _RecordingSession()
    repository = SourceCatalogRepository(lambda: _SessionContext(session))

    with pytest.raises(SourceOwnershipConflict):
        asyncio.run(repository.adopt_source('user-1', 'source-1'))


def test_find_similar_blocks_uses_neo4j_vector_index_and_omits_embedding():
    session = _RecordingSession(
        _RecordingResult(
            [
                {
                    'uuid': 'nearest-block',
                    'block_type': 'paragraph',
                    'content': 'nearest',
                    'score': 0.99,
                },
                {
                    'uuid': 'next-block',
                    'block_type': 'equation',
                    'content': None,
                    'score': 0.81,
                },
            ]
        )
    )
    repository = SourceBlockRepository(lambda: _SessionContext(session))

    results = asyncio.run(
        repository.find_similar_blocks('query-block', top_k=2)
    )

    assert [result.uuid for result in results] == [
        'nearest-block',
        'next-block',
    ]
    assert results[0].score == 0.99
    assert 'embedding' not in results[0].model_dump()
