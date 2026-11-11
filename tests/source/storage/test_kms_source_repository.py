import asyncio
from datetime import UTC, datetime

import pytest

from kms.source.models.material import Instruction
from kms.source.models.snapshot import (
    SourceSnapshot,
    processed_snapshot_fingerprint,
)
from kms.source.models.source import Source, SourceBlock, SourcePage
from kms.source.storage.catalog import (
    SourceCatalogRepository,
    SourceOwnershipConflict,
)
from kms.source.storage.records import (
    source_graph_parameters,
    source_pages_from_records,
)


class _RecordingResult:
    def __init__(self, rows: list[dict[str, object]] | None = None) -> None:
        self._rows = rows or []

    async def consume(self) -> None:
        return None

    async def data(self) -> list[dict[str, object]]:
        return self._rows

    async def single(self, *, strict: bool = False) -> dict[str, object] | None:
        if strict:
            assert len(self._rows) == 1
        return self._rows[0] if self._rows else None


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


def test_load_pages_restores_order_and_direct_image_provenance():
    created_at = datetime(2026, 1, 2, tzinfo=UTC)
    updated_at = datetime(2026, 1, 3, tzinfo=UTC)
    invalidated_at = datetime(2026, 1, 4, tzinfo=UTC)
    records = [
        {
            'index': 0,
            'markdown': 'first page',
            'blocks': [
                {
                    'uuid': 'block-1',
                    'created_at': created_at,
                    'updated_at': updated_at,
                    'invalidated_at': invalidated_at,
                    'block_type': 'image',
                    'content': 'figure description',
                    'embedding': [0.0, 1.0],
                    'crop_path': 'crop.png',
                    'crop_sha256': 'crop-digest',
                    'crop_bbox': [1, 2, 3, 4],
                    'image_path': 'figure.png',
                    'image_sha256': 'image-digest',
                    'image_page_indexes': [0, 2],
                }
            ],
        },
        {
            'index': 1,
            'markdown': 'empty page',
            'created_at': created_at,
            'updated_at': created_at,
            'blocks': [],
        },
    ]
    pages = source_pages_from_records(records)
    expected_pages = [
        SourcePage(
            index=0,
            markdown='first page',
            blocks=[
                SourceBlock(
                    uuid='block-1',
                    created_at=created_at,
                    updated_at=updated_at,
                    invalidated_at=invalidated_at,
                    block_type='image',
                    content='figure description',
                    embedding=[0.0, 1.0],
                    crop_path='crop.png',
                    crop_sha256='crop-digest',
                    crop_bbox=(1, 2, 3, 4),
                    image_path='figure.png',
                    image_sha256='image-digest',
                    image_page_indexes=[0, 2],
                )
            ],
        ),
        SourcePage(index=1, markdown='empty page', blocks=[]),
    ]
    assert pages == expected_pages
    snapshot = SourceSnapshot(
        source=Source(uuid='source-1', key='source'), pages=pages
    )
    expected = snapshot.model_copy(update={'pages': expected_pages})
    assert processed_snapshot_fingerprint(
        snapshot
    ) == processed_snapshot_fingerprint(expected)


def test_list_sources_returns_stable_user_source_summaries():
    created_at = datetime(2026, 1, 2, tzinfo=UTC)
    updated_at = datetime(2026, 1, 3, tzinfo=UTC)
    invalidated_at = datetime(2026, 1, 4, tzinfo=UTC)
    session = _RecordingSession(
        _RecordingResult(
            [
                {
                    'uuid': 'source-2',
                    'key': 'second.pdf',
                    'created_at': created_at,
                    'updated_at': updated_at,
                    'invalidated_at': invalidated_at,
                    'revision': 2,
                },
                {
                    'uuid': 'source-1',
                    'key': 'first.pdf',
                    'created_at': created_at,
                    'revision': 0,
                    'updated_at': created_at,
                    'invalidated_at': None,
                },
            ]
        )
    )

    sources = asyncio.run(
        SourceCatalogRepository(lambda: _SessionContext(session)).list_sources(
            'user-1'
        )
    )

    assert sources == [
        Source(
            uuid='source-2',
            key='second.pdf',
            created_at=created_at,
            updated_at=updated_at,
            invalidated_at=invalidated_at,
            revision=2,
        ),
        Source(
            uuid='source-1',
            key='first.pdf',
            created_at=created_at,
            revision=0,
        ),
    ]


def test_list_unowned_sources_excludes_owned_source_catalog():
    session = _RecordingSession(
        _RecordingResult(
            [
                {
                    'uuid': 'legacy-1',
                    'key': 'legacy.pdf',
                    'created_at': datetime(2026, 1, 2, tzinfo=UTC),
                    'updated_at': datetime(2026, 1, 2, tzinfo=UTC),
                    'invalidated_at': None,
                    'revision': 0,
                }
            ]
        )
    )

    sources = asyncio.run(
        SourceCatalogRepository(
            lambda: _SessionContext(session)
        ).list_unowned_sources()
    )

    assert sources == [
        Source(
            uuid='legacy-1',
            key='legacy.pdf',
            created_at=datetime(2026, 1, 2, tzinfo=UTC),
            updated_at=datetime(2026, 1, 2, tzinfo=UTC),
            revision=0,
        )
    ]


def test_adopt_source_locks_source_and_rejects_claimed_source():
    session = _RecordingSession()
    repository = SourceCatalogRepository(lambda: _SessionContext(session))

    with pytest.raises(SourceOwnershipConflict):
        asyncio.run(repository.adopt_source('user-1', 'source-1'))


def test_source_graph_parameters_preserve_vertex_lifecycle_times():
    created_at = datetime(2026, 1, 2, tzinfo=UTC)
    updated_at = datetime(2026, 1, 3, tzinfo=UTC)
    invalidated_at = datetime(2026, 1, 4, tzinfo=UTC)
    snapshot = SourceSnapshot(
        source=Source(
            uuid='source-1',
            key='source',
            created_at=created_at,
            updated_at=updated_at,
            invalidated_at=invalidated_at,
        ),
        pages=[
            SourcePage(
                index=0,
                blocks=[
                    SourceBlock(
                        uuid='block-1',
                        block_type='paragraph',
                        content='content',
                        created_at=created_at,
                        updated_at=updated_at,
                        invalidated_at=invalidated_at,
                    )
                ],
            )
        ],
        instructions=[
            Instruction(
                uuid='instruction-1',
                created_at=created_at,
                updated_at=updated_at,
                invalidated_at=invalidated_at,
                member_block_uuids=['block-1'],
            )
        ],
    )

    parameters = source_graph_parameters(snapshot)

    assert parameters['source']['created_at'] is created_at
    assert parameters['source']['updated_at'] is updated_at
    assert parameters['source']['invalidated_at'] is invalidated_at
    assert parameters['blocks'][0]['created_at'] is created_at
    assert parameters['blocks'][0]['updated_at'] is updated_at
    assert parameters['blocks'][0]['invalidated_at'] is invalidated_at
    assert parameters['instructions'][0]['updated_at'] is updated_at
    assert parameters['instructions'][0]['invalidated_at'] is invalidated_at
    assert isinstance(parameters['edge_created_at'], datetime)
    changed_timestamp_snapshot = snapshot.model_copy(
        update={
            'source': snapshot.source.model_copy(
                update={
                    'created_at': datetime(2026, 1, 5, tzinfo=UTC),
                    'updated_at': datetime(2026, 1, 6, tzinfo=UTC),
                    'invalidated_at': datetime(2026, 1, 7, tzinfo=UTC),
                }
            ),
            'pages': [
                snapshot.pages[0].model_copy(
                    update={
                        'blocks': [
                            snapshot.pages[0]
                            .blocks[0]
                            .model_copy(
                                update={
                                    'created_at': datetime(
                                        2026, 1, 5, tzinfo=UTC
                                    ),
                                    'updated_at': datetime(
                                        2026, 1, 6, tzinfo=UTC
                                    ),
                                    'invalidated_at': datetime(
                                        2026, 1, 7, tzinfo=UTC
                                    ),
                                }
                            )
                        ]
                    }
                )
            ],
        }
    )
    assert processed_snapshot_fingerprint(
        snapshot
    ) == processed_snapshot_fingerprint(changed_timestamp_snapshot)


def test_default_vertex_updated_at_copies_created_at():
    created_at = datetime(2026, 1, 2, tzinfo=UTC)

    source = Source(uuid='source-1', key='source', created_at=created_at)
    block = SourceBlock(
        uuid='block-1',
        block_type='paragraph',
        content='content',
        created_at=created_at,
    )
    instruction = Instruction(created_at=created_at)

    assert source.updated_at is created_at
    assert block.updated_at is created_at
    assert instruction.updated_at is created_at
