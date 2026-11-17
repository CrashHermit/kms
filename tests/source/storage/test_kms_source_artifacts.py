import hashlib
from pathlib import Path

import pytest

from kms.artifacts import ArtifactIntegrityError
from kms.core.models import Vertex
from kms.source.models.snapshot import (
    SourceSnapshot,
    SourceSnapshotConflict,
    assert_matching_snapshot,
    assert_source_embedding_profile,
    finalize_snapshot_identities,
    processed_snapshot_fingerprint,
    require_source_schema,
)
from kms.source.models.source import BlockType, Source, SourceBlock
from kms.source.models.unit import SourceUnit, SourceUnitKind
from kms.source.storage.artifacts import SourceArtifactStore, read_image_bytes


def _source() -> Source:
    return Source(
        uuid='00000000-0000-0000-0000-000000000001',
        key='source',
        metadata={},
    )


def _snapshot(source: Source, units: list[SourceUnit]) -> SourceSnapshot:
    return SourceSnapshot(source=source, units=units)


def test_finalize_units_hashes_and_deduplicates_shared_media(tmp_path):
    crop = tmp_path / 'crop.jpg'
    image = tmp_path / 'figure.png'
    crop.write_bytes(b'crop bytes')
    image.write_bytes(b'png bytes')
    blocks = [
        SourceBlock(
            uuid='text',
            block_type=BlockType.PARAGRAPH,
            content='A theorem.',
            crop_path=str(crop),
        ),
        SourceBlock(
            uuid='figure',
            block_type=BlockType.IMAGE,
            content='A figure.',
            image_path=str(image),
            crop_path=str(crop),
        ),
    ]
    units = [
        SourceUnit(kind=SourceUnitKind.CONTENT, blocks=blocks),
        SourceUnit(
            kind=SourceUnitKind.EXERCISE,
            blocks=[blocks[0].model_copy(update={'uuid': 'copy'})],
        ),
    ]

    durable = SourceArtifactStore(tmp_path / 'durable').finalize_units(units)
    text, figure = durable[0].blocks
    assert text.crop_sha256 == hashlib.sha256(b'crop bytes').hexdigest()
    assert Path(text.crop_path).read_bytes() == b'crop bytes'
    assert figure.crop_path == text.crop_path
    assert durable[1].blocks[0].crop_path == text.crop_path
    assert figure.image_sha256 == hashlib.sha256(b'png bytes').hexdigest()
    assert (
        read_image_bytes(figure.image_path, figure.image_sha256) == b'png bytes'
    )


def test_processed_fingerprint_and_ids_ignore_provisional_identity_and_paths(
    tmp_path,
):
    image = tmp_path / 'image.png'
    image.write_bytes(b'png')
    digest = hashlib.sha256(b'png').hexdigest()
    source = _source()
    created_at = Vertex().created_at
    block = SourceBlock(
        uuid='provisional-a',
        block_type=BlockType.IMAGE,
        content='Figure description',
        embedding=[1.0, 2.0],
        image_path=str(image),
        image_sha256=digest,
        crop_path='/one/crop.png',
        crop_sha256='f' * 64,
        created_at=created_at,
    )
    unit = SourceUnit(
        uuid='unit-a', kind=SourceUnitKind.CONTENT, blocks=[block]
    )
    relocated = SourceUnit(
        uuid='unit-b',
        kind=SourceUnitKind.CONTENT,
        blocks=[
            block.model_copy(
                update={
                    'uuid': 'provisional-b',
                    'image_path': '/different/image.png',
                    'crop_path': '/different/crop.png',
                    'embedding': [9.0, 8.0],
                }
            )
        ],
    )
    assert processed_snapshot_fingerprint(
        _snapshot(source, [unit])
    ) == processed_snapshot_fingerprint(_snapshot(source, [relocated]))
    finalized = finalize_snapshot_identities(source, [unit])
    relocated_final = finalize_snapshot_identities(source, [relocated])
    assert finalized.source.metadata['identity_schema'] == 'source-content-v5'
    assert finalized.units[0].uuid == relocated_final.units[0].uuid
    assert (
        finalized.units[0].blocks[0].uuid
        == relocated_final.units[0].blocks[0].uuid
    )
    assert finalized.units[0].created_at == unit.created_at
    assert finalized.units[0].blocks[0].created_at == created_at

    changed = _snapshot(
        source,
        [
            unit.model_copy(
                update={
                    'blocks': [
                        block.model_copy(update={'image_sha256': '0' * 64})
                    ]
                }
            )
        ],
    )
    assert processed_snapshot_fingerprint(
        _snapshot(source, [unit])
    ) != processed_snapshot_fingerprint(changed)
    assert (
        finalized.units[0].blocks[0].uuid
        != finalize_snapshot_identities(source, changed.units)
        .units[0]
        .blocks[0]
        .uuid
    )


def test_fingerprint_covers_kind_order_and_content():
    source = _source()
    first = SourceBlock(
        block_type=BlockType.PARAGRAPH,
        content='one',
    )
    second = SourceBlock(
        block_type=BlockType.PARAGRAPH,
        content='two',
    )
    content = SourceUnit(kind=SourceUnitKind.CONTENT, blocks=[first, second])
    exercise = SourceUnit(
        kind=SourceUnitKind.EXERCISE,
        blocks=[first.model_copy(update={'content': 'task'})],
    )
    baseline = processed_snapshot_fingerprint(
        _snapshot(source, [content, exercise])
    )
    assert baseline != processed_snapshot_fingerprint(
        _snapshot(source, [exercise, content])
    )
    assert baseline != processed_snapshot_fingerprint(
        _snapshot(
            source,
            [
                content.model_copy(update={'blocks': [second, first]}),
                exercise,
            ],
        )
    )
    assert baseline != processed_snapshot_fingerprint(
        _snapshot(
            source,
            [
                content.model_copy(update={'kind': SourceUnitKind.EXERCISE}),
                exercise,
            ],
        )
    )
    assert baseline != processed_snapshot_fingerprint(
        _snapshot(
            source,
            [
                content.model_copy(
                    update={
                        'blocks': [
                            first.model_copy(update={'content': 'changed'}),
                            second,
                        ]
                    }
                ),
                exercise,
            ],
        )
    )


def test_complete_snapshot_matching_detects_content_conflict():
    source = _source()
    unit = SourceUnit(
        kind=SourceUnitKind.CONTENT,
        blocks=[SourceBlock(block_type=BlockType.PARAGRAPH, content='one')],
    )
    candidate = finalize_snapshot_identities(source, [unit])
    persisted = candidate.model_copy(
        update={'source': candidate.source.model_copy(update={'revision': 1})}
    )
    assert_matching_snapshot(candidate, persisted)
    changed = persisted.model_copy(
        update={
            'units': [
                candidate.units[0].model_copy(
                    update={
                        'blocks': [
                            candidate.units[0]
                            .blocks[0]
                            .model_copy(update={'content': 'changed'})
                        ]
                    }
                )
            ]
        }
    )
    with pytest.raises(SourceSnapshotConflict):
        assert_matching_snapshot(candidate, changed)


def test_duplicate_block_occurrences_get_distinct_stable_ids():
    source = _source()
    block = SourceBlock(
        block_type=BlockType.PARAGRAPH,
        content='shared instruction',
    )
    units = [
        SourceUnit(kind=SourceUnitKind.EXERCISE, blocks=[block.model_copy()]),
        SourceUnit(kind=SourceUnitKind.EXERCISE, blocks=[block.model_copy()]),
    ]
    finalized = finalize_snapshot_identities(source, units)
    repeated = finalize_snapshot_identities(
        source,
        [
            units[0].model_copy(update={'uuid': 'different-unit'}),
            units[1].model_copy(update={'uuid': 'different-unit-2'}),
        ],
    )
    ids = [unit.blocks[0].uuid for unit in finalized.units]
    assert ids[0] != ids[1]
    assert ids == [unit.blocks[0].uuid for unit in repeated.units]


def test_processed_fingerprint_rejects_unfinalized_media_digests():
    source = _source()
    image = SourceBlock(
        content=None,
        block_type=BlockType.IMAGE,
        image_path='image.png',
    )
    with pytest.raises(
        ValueError, match='Source media digests must be finalized'
    ):
        processed_snapshot_fingerprint(
            _snapshot(
                source,
                [SourceUnit(kind=SourceUnitKind.CONTENT, blocks=[image])],
            )
        )


def test_image_read_rejects_changed_bytes_and_missing_digest(tmp_path):
    path = tmp_path / 'image.png'
    path.write_bytes(b'original')
    digest = hashlib.sha256(b'original').hexdigest()
    assert read_image_bytes(str(path), digest) == b'original'
    path.write_bytes(b'changed')
    with pytest.raises(
        ValueError,
        match='Source image content does not match its recorded digest',
    ):
        read_image_bytes(str(path), digest)


def test_corrupted_content_addressed_file_fails_instead_of_overwrite(tmp_path):
    store = SourceArtifactStore(tmp_path)
    path = store.write(b'correct', suffix='.png')
    path.write_bytes(b'corrupt')
    with pytest.raises(ArtifactIntegrityError):
        store.write(b'correct', suffix='.png')


def test_source_embedding_profile_checks_unit_and_readable_blocks():
    snapshot = SourceSnapshot(
        source=Source(
            uuid='source',
            key='source',
            metadata={
                'embedding_profile': {'model': 'model-a', 'dimension': 2}
            },
        ),
        units=[
            SourceUnit(
                kind=SourceUnitKind.CONTENT,
                embedding=[1.0, 2.0],
                blocks=[
                    SourceBlock(
                        block_type=BlockType.MARKDOWN,
                        content='Text',
                        embedding=[1.0, 2.0],
                    ),
                    SourceBlock(
                        block_type=BlockType.IMAGE,
                        content=None,
                        image_path='/image.png',
                        embedding=[1.0, 2.0],
                    ),
                    SourceBlock(block_type=BlockType.PARAGRAPH, content='  '),
                ],
            ),
            SourceUnit(
                kind=SourceUnitKind.CONTENT,
                blocks=[
                    SourceBlock(block_type=BlockType.PARAGRAPH, content='')
                ],
            ),
        ],
    )
    assert_source_embedding_profile(snapshot, model='model-a', dimension=2)
    with pytest.raises(ValueError, match='Source embedding profile mismatch'):
        assert_source_embedding_profile(snapshot, model='model-b', dimension=2)


def test_source_schema_guard_rejects_v3_records():
    require_source_schema('source-content-v5')
    with pytest.raises(
        ValueError, match='Source format is obsolete; re-ingest the source'
    ):
        require_source_schema('source-content-v3')
