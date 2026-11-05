import hashlib
from pathlib import Path

import pytest

from kms.core.model.block import SourceBlock
from kms.core.model.block_types import BlockType
from kms.core.model.page import SourcePage
from kms.core.model.source import Source
from kms.core.model.source_processing.admission import SourceSnapshot
from kms.core.model.visual_asset import VisualAsset
from kms.core.source_snapshot import (
    SourceSnapshotConflict,
    assert_matching_snapshot,
    assert_source_embedding_profile,
    finalize_snapshot_identities,
    processed_snapshot_fingerprint,
    source_topology_fingerprint,
)
from kms.service.source_artifacts import SourceArtifactStore, read_asset_bytes


def test_finalize_pages_hashes_actual_media_and_preserves_occurrences(tmp_path):
    crop = tmp_path / 'crop.jpg'
    image = tmp_path / 'figure.png'
    crop.write_bytes(b'crop bytes')
    image.write_bytes(b'png bytes')
    pages = [
        SourcePage(
            index=4,
            markdown='source page',
            blocks=[
                SourceBlock(
                    uuid='block',
                    block_type=BlockType.MARKDOWN,
                    content='A theorem.',
                    crop_path=str(crop),
                    crop_bbox=(1, 2, 3, 4),
                    assets=[
                        VisualAsset(uuid='asset-a', path=str(image)),
                        VisualAsset(uuid='asset-b', path=str(image)),
                    ],
                )
            ],
        )
    ]

    durable = SourceArtifactStore(tmp_path / 'durable').finalize_pages(pages)
    block = durable[0].blocks[0]

    assert block.crop_sha256 == hashlib.sha256(b'crop bytes').hexdigest()
    assert Path(block.crop_path).read_bytes() == b'crop bytes'
    assert [asset.uuid for asset in block.assets] == ['asset-a', 'asset-b']
    assert [asset.content_sha256 for asset in block.assets] == [
        hashlib.sha256(b'png bytes').hexdigest(),
        hashlib.sha256(b'png bytes').hexdigest(),
    ]
    assert block.assets[0].path == block.assets[1].path
    assert read_asset_bytes(block.assets[0]) == b'png bytes'


def test_processed_identity_uses_content_digests_not_vectors_or_paths():
    digest = hashlib.sha256(b'asset').hexdigest()
    crop_digest = hashlib.sha256(b'crop').hexdigest()
    original = [
        SourcePage(
            index=2,
            markdown='page',
            blocks=[
                SourceBlock(
                    block_type=BlockType.MARKDOWN,
                    content='Text',
                    embedding=[1.0, 2.0],
                    crop_path='/one/crop.png',
                    crop_sha256=crop_digest,
                    crop_bbox=(0, 1, 2, 3),
                    assets=[
                        VisualAsset(
                            uuid='first',
                            path='/one/image.png',
                            content_sha256=digest,
                        )
                    ],
                )
            ],
        )
    ]
    relocated = [
        original[0].model_copy(
            update={
                'blocks': [
                    original[0]
                    .blocks[0]
                    .model_copy(
                        update={
                            'embedding': [9.0, 8.0],
                            'crop_path': '/two/crop.png',
                            'assets': [
                                original[0]
                                .blocks[0]
                                .assets[0]
                                .model_copy(update={'path': '/two/image.png'})
                            ],
                        }
                    )
                ]
            }
        )
    ]

    assert processed_snapshot_fingerprint(original) == (
        processed_snapshot_fingerprint(relocated)
    )

    source = Source(
        uuid='00000000-0000-0000-0000-000000000001',
        key='source',
    )
    first = finalize_snapshot_identities(source, original)
    second = finalize_snapshot_identities(source, relocated)
    assert (
        first.source.metadata['processed_digest']
        == (second.source.metadata['processed_digest'])
    )
    assert first.pages[0].blocks[0].uuid == second.pages[0].blocks[0].uuid
    assert first.pages[0].blocks[0].assets[0].uuid == (
        second.pages[0].blocks[0].assets[0].uuid
    )
    changed = relocated[0].model_copy(
        update={
            'blocks': [
                relocated[0]
                .blocks[0]
                .model_copy(
                    update={
                        'assets': [
                            VisualAsset(
                                path='/two/image.png', content_sha256='0' * 64
                            )
                        ]
                    }
                )
            ]
        }
    )
    assert processed_snapshot_fingerprint(original) != (
        processed_snapshot_fingerprint([changed])
    )


def test_split_topology_does_not_replace_capture_identity():
    source = Source(
        uuid='00000000-0000-0000-0000-000000000001',
        key='source',
        metadata={'normalization_block_positions': {'root': 0}},
    )
    original_pages = [
        SourcePage(
            index=0,
            markdown='Two exercises.',
            blocks=[
                SourceBlock(
                    uuid='root',
                    block_type=BlockType.MARKDOWN,
                    content='Two exercises.',
                )
            ],
        )
    ]
    captured = finalize_snapshot_identities(source, original_pages)
    parent_uuid = captured.pages[0].blocks[0].uuid
    split_pages = [
        SourcePage(
            index=0,
            markdown='Two exercises.',
            blocks=[
                SourceBlock(
                    uuid='child-1',
                    block_type=BlockType.MARKDOWN,
                    content='Exercise one.',
                    origin_uuid=parent_uuid,
                ),
                SourceBlock(
                    uuid='child-2',
                    block_type=BlockType.MARKDOWN,
                    content='Exercise two.',
                    origin_uuid=parent_uuid,
                ),
            ],
        )
    ]
    processed = captured.model_copy(
        update={
            'source': captured.source.model_copy(
                update={
                    'revision': 1,
                    'metadata': {
                        **captured.source.metadata,
                        'normalization_block_positions': {
                            'child-1': 0,
                            'child-2': 0,
                        },
                        'pedagogical_topology_digest': (
                            source_topology_fingerprint(split_pages)
                        ),
                    },
                }
            ),
            'pages': split_pages,
        }
    )
    assert source_topology_fingerprint(captured.pages) != (
        source_topology_fingerprint(split_pages)
    )

    assert_matching_snapshot(captured, processed)
    altered_candidate = captured.model_copy(update={'pages': split_pages})
    with pytest.raises(SourceSnapshotConflict):
        assert_matching_snapshot(altered_candidate, processed)
    altered_persisted = processed.model_copy(update={'pages': captured.pages})
    with pytest.raises(SourceSnapshotConflict):
        assert_matching_snapshot(captured, altered_persisted)


def test_processed_identity_rejects_unfinalized_media_digests():
    page = SourcePage(
        index=0,
        blocks=[
            SourceBlock(
                block_type=BlockType.MARKDOWN,
                content='Text',
                assets=[VisualAsset(path='image.png')],
            )
        ],
    )
    with pytest.raises(
        ValueError,
        match='Source media digests must be finalized before fingerprinting',
    ):
        processed_snapshot_fingerprint([page])

    crop_only_page = SourcePage(
        index=0,
        blocks=[
            SourceBlock(
                block_type=BlockType.MARKDOWN,
                content='Text',
                crop_path='crop.png',
            )
        ],
    )
    with pytest.raises(
        ValueError,
        match='Source media digests must be finalized before fingerprinting',
    ):
        processed_snapshot_fingerprint([crop_only_page])


def test_asset_read_rejects_changed_bytes_and_missing_digest(tmp_path):
    path = tmp_path / 'asset.png'
    path.write_bytes(b'original')
    asset = VisualAsset(
        path=str(path),
        content_sha256=hashlib.sha256(b'original').hexdigest(),
    )
    assert read_asset_bytes(asset) == b'original'
    path.write_bytes(b'changed')
    with pytest.raises(
        ValueError,
        match='Source asset content does not match its recorded digest',
    ):
        read_asset_bytes(asset)
    with pytest.raises(
        ValueError,
        match='Source asset content does not match its recorded digest',
    ):
        read_asset_bytes(VisualAsset(path=str(path)))


def test_corrupted_content_addressed_file_fails_instead_of_overwrite(tmp_path):
    store = SourceArtifactStore(tmp_path)
    path = store.store(b'correct', suffix='.png')
    path.write_bytes(b'corrupt')

    with pytest.raises(
        ValueError,
        match='Stored source artifact content does not match its digest',
    ):
        store.store(b'correct', suffix='.png')


def test_source_embedding_profile_checks_text_vectors_and_ignores_textless():
    snapshot = SourceSnapshot(
        source=Source(
            uuid='source',
            key='source',
            metadata={
                'embedding_profile': {'model': 'model-a', 'dimension': 2}
            },
        ),
        pages=[
            SourcePage(
                index=0,
                blocks=[
                    SourceBlock(
                        block_type=BlockType.MARKDOWN,
                        content='Text',
                        embedding=[1.0, 2.0],
                    ),
                    SourceBlock(
                        block_type=BlockType.IMAGE,
                        content=None,
                        embedding=None,
                    ),
                ],
            )
        ],
    )

    assert_source_embedding_profile(snapshot, model='model-a', dimension=2)
    with pytest.raises(
        ValueError,
        match='Source embedding profile mismatch; reindex required',
    ):
        assert_source_embedding_profile(snapshot, model='model-b', dimension=2)


@pytest.mark.parametrize('vector', [None, [1.0], [1.0, float('nan')]])
def test_source_embedding_profile_rejects_invalid_text_vectors(vector):
    block = SourceBlock(
        block_type=BlockType.MARKDOWN,
        content='Text',
        embedding=vector,
    )
    snapshot = SourceSnapshot(
        source=Source(
            uuid='source',
            key='source',
            metadata={
                'embedding_profile': {'model': 'model-a', 'dimension': 2}
            },
        ),
        pages=[SourcePage(index=0, blocks=[block])],
    )

    with pytest.raises(
        ValueError,
        match='Source embedding profile mismatch; reindex required',
    ):
        assert_source_embedding_profile(snapshot, model='model-a', dimension=2)
