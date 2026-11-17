import asyncio
import json
from datetime import UTC, datetime

from kms.source.models.snapshot import (
    SourceSnapshot,
    processed_snapshot_fingerprint,
)
from kms.source.models.source import Source, SourceBlock
from kms.source.models.unit import SourceUnit, SourceUnitKind
from kms.source.storage.catalog import (
    SourceCatalogRepository,
)
from kms.source.storage.records import (
    source_graph_parameters,
    source_units_from_records,
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


def _record(
    uuid: str, kind: str, blocks: list[dict[str, object]]
) -> dict[str, object]:
    created_at = datetime(2026, 1, 2, tzinfo=UTC)
    return {
        'uuid': uuid,
        'created_at': created_at,
        'updated_at': created_at,
        'invalidated_at': None,
        'kind': kind,
        'embedding': [0.0, 1.0],
        'blocks': blocks,
    }


def _block_record(uuid: str, content: str) -> dict[str, object]:
    created_at = datetime(2026, 1, 2, tzinfo=UTC)
    return {
        'uuid': uuid,
        'created_at': created_at,
        'updated_at': created_at,
        'invalidated_at': None,
        'block_type': 'paragraph',
        'content': content,
        'embedding': [0.0, 1.0],
        'crop_path': None,
        'crop_sha256': None,
        'crop_bbox': None,
        'image_path': None,
        'image_sha256': None,
    }


def test_unit_hydration_preserves_order_and_lifecycle():
    created_at = datetime(2026, 1, 2, tzinfo=UTC)
    updated_at = datetime(2026, 1, 3, tzinfo=UTC)
    invalidated_at = datetime(2026, 1, 4, tzinfo=UTC)
    records = [
        {
            'uuid': 'unit-exercise',
            'created_at': created_at,
            'updated_at': updated_at,
            'invalidated_at': invalidated_at,
            'kind': 'exercise',
            'embedding': [0.0, 1.0],
            'blocks': [
                {
                    **_block_record('block-1', 'first'),
                    'updated_at': updated_at,
                    'invalidated_at': invalidated_at,
                    'block_type': 'image',
                    'content': 'figure description',
                    'crop_path': 'crop.png',
                    'crop_sha256': 'crop-digest',
                    'crop_bbox': [1, 2, 3, 4],
                    'image_path': 'figure.png',
                    'image_sha256': 'image-digest',
                },
                _block_record('block-2', 'second'),
            ],
        },
        _record('unit-content', 'content', [_block_record('block-3', 'later')]),
    ]
    units = source_units_from_records(records)
    assert units[0] == SourceUnit(
        uuid='unit-exercise',
        created_at=created_at,
        updated_at=updated_at,
        invalidated_at=invalidated_at,
        kind=SourceUnitKind.EXERCISE,
        embedding=[0.0, 1.0],
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
            ),
            SourceBlock(
                uuid='block-2',
                created_at=created_at,
                block_type='paragraph',
                content='second',
                embedding=[0.0, 1.0],
            ),
        ],
    )
    assert units[1].kind is SourceUnitKind.CONTENT
    assert [block.content for block in units[1].blocks] == ['later']
    assert source_units_from_records([]) == []


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

    assert [source.uuid for source in sources] == ['source-2', 'source-1']
    assert sources[0].invalidated_at == invalidated_at
    assert sources[1].revision == 0


def test_source_graph_parameters_preserve_nested_order_and_lifecycle_times():
    created_at = datetime(2026, 1, 2, tzinfo=UTC)
    updated_at = datetime(2026, 1, 3, tzinfo=UTC)
    invalidated_at = datetime(2026, 1, 4, tzinfo=UTC)
    blocks = [
        SourceBlock(
            uuid=f'block-{index}',
            block_type='paragraph',
            content=f'content {index}',
            created_at=created_at,
            updated_at=updated_at,
            invalidated_at=invalidated_at,
        )
        for index in range(2)
    ]
    unit = SourceUnit(
        uuid='unit-1',
        kind=SourceUnitKind.EXERCISE,
        blocks=blocks,
        created_at=created_at,
        updated_at=updated_at,
        invalidated_at=invalidated_at,
    )
    snapshot = SourceSnapshot(
        source=Source(
            uuid='source-1',
            key='source',
            created_at=created_at,
            updated_at=updated_at,
            invalidated_at=invalidated_at,
            metadata={'identity_schema': 'source-content-v5'},
        ),
        units=[unit],
    )
    parameters = source_graph_parameters(snapshot)
    assert 'page_indexes' not in parameters['blocks'][0]
    assert 'page_count' not in json.loads(parameters['source']['metadata'])
    assert 'page_indexes' not in json.loads(parameters['source']['metadata'])
    assert parameters['source']['created_at'] is created_at
    assert parameters['units'][0]['kind'] == 'exercise'
    assert parameters['units'][0]['created_at'] is created_at
    assert parameters['units'][0]['updated_at'] is updated_at
    assert parameters['units'][0]['invalidated_at'] is invalidated_at
    assert parameters['blocks'][0]['created_at'] is created_at
    assert parameters['blocks'][0]['updated_at'] is updated_at
    assert parameters['blocks'][0]['invalidated_at'] is invalidated_at
    assert [block['uuid'] for block in parameters['blocks']] == [
        'block-0',
        'block-1',
    ]
    assert [pair['block_uuid'] for pair in parameters['unit_block_pairs']] == [
        'block-0',
        'block-1',
    ]
    assert parameters['block_pairs'] == [
        {'unit_uuid': 'unit-1', 'from_uuid': 'block-0', 'to_uuid': 'block-1'}
    ]
    assert parameters['unit_bounds'] == [
        {
            'unit_uuid': 'unit-1',
            'first_block_uuid': 'block-0',
            'last_block_uuid': 'block-1',
        }
    ]
    assert isinstance(parameters['edge_created_at'], datetime)
    changed = snapshot.model_copy(
        update={
            'source': snapshot.source.model_copy(
                update={
                    'created_at': datetime(2026, 1, 5, tzinfo=UTC),
                    'updated_at': datetime(2026, 1, 6, tzinfo=UTC),
                    'invalidated_at': datetime(2026, 1, 7, tzinfo=UTC),
                }
            )
        }
    )
    assert processed_snapshot_fingerprint(
        snapshot
    ) == processed_snapshot_fingerprint(changed)
