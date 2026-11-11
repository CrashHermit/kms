import hashlib
from pathlib import Path

import pytest

from kms.artifacts import ArtifactIntegrityError
from kms.source.models.material import (
    Instruction,
    ProcedureDraft,
    StatementDraft,
)
from kms.source.models.snapshot import (
    SourceSnapshot,
    SourceSnapshotConflict,
    assert_matching_snapshot,
    assert_source_embedding_profile,
    finalize_snapshot_identities,
    processed_snapshot_fingerprint,
    require_source_schema,
)
from kms.source.models.source import BlockType, Source, SourceBlock, SourcePage
from kms.source.storage.artifacts import SourceArtifactStore, read_image_bytes


def _snapshot(source: Source, pages: list[SourcePage]) -> SourceSnapshot:
    return SourceSnapshot(source=source, pages=pages)


def test_finalize_pages_hashes_and_deduplicates_direct_media(tmp_path):
    crop = tmp_path / 'crop.jpg'
    image = tmp_path / 'figure.png'
    crop.write_bytes(b'crop bytes')
    image.write_bytes(b'png bytes')
    page = SourcePage(
        index=4,
        markdown='source page',
        blocks=[
            SourceBlock(
                uuid='text',
                block_type=BlockType.MARKDOWN,
                content='A theorem.',
                crop_path=str(crop),
                crop_bbox=(1, 2, 3, 4),
            ),
            SourceBlock(
                uuid='figure',
                block_type=BlockType.IMAGE,
                content='A figure.',
                image_path=str(image),
                image_page_indexes=[4],
                crop_path=str(crop),
            ),
        ],
    )
    durable = SourceArtifactStore(tmp_path / 'durable').finalize_pages([page])
    text, figure = durable[0].blocks
    assert text.crop_sha256 == hashlib.sha256(b'crop bytes').hexdigest()
    assert Path(text.crop_path).read_bytes() == b'crop bytes'
    assert figure.crop_path == text.crop_path
    assert figure.image_sha256 == hashlib.sha256(b'png bytes').hexdigest()
    assert (
        read_image_bytes(figure.image_path, figure.image_sha256) == b'png bytes'
    )


def test_processed_fingerprint_ignores_paths_and_vectors_but_hashes_image(
    tmp_path,
):
    image = tmp_path / 'image.png'
    image.write_bytes(b'png')
    digest = hashlib.sha256(b'png').hexdigest()
    source = Source(uuid='00000000-0000-0000-0000-000000000001', key='source')
    page = SourcePage(
        index=0,
        markdown='page',
        blocks=[
            SourceBlock(
                uuid='one',
                block_type=BlockType.IMAGE,
                content='Figure description',
                embedding=[1.0, 2.0],
                image_path=str(image),
                image_sha256=digest,
                image_page_indexes=[0],
                crop_path='/one/crop.png',
                crop_sha256='f' * 64,
            )
        ],
    )
    snapshot = _snapshot(source, [page])
    relocated = _snapshot(
        source,
        [
            page.model_copy(
                update={
                    'blocks': [
                        page.blocks[0].model_copy(
                            update={
                                'image_path': '/different/image.png',
                                'crop_path': '/different/crop.png',
                                'embedding': [9.0, 8.0],
                            }
                        )
                    ]
                }
            )
        ],
    )
    assert processed_snapshot_fingerprint(
        snapshot
    ) == processed_snapshot_fingerprint(relocated)
    finalized = finalize_snapshot_identities(source, [page], [], [], [])
    assert finalized.source.metadata['identity_schema'] == 'source-content-v3'
    assert (
        finalized.pages[0].blocks[0].uuid
        == finalize_snapshot_identities(
            source, [relocated.pages[0]], [], [], []
        )
        .pages[0]
        .blocks[0]
        .uuid
    )
    changed = _snapshot(
        source,
        [
            page.model_copy(
                update={
                    'blocks': [
                        page.blocks[0].model_copy(
                            update={'image_sha256': '0' * 64}
                        )
                    ]
                }
            )
        ],
    )
    assert processed_snapshot_fingerprint(
        snapshot
    ) != processed_snapshot_fingerprint(changed)


def test_complete_snapshot_fingerprint_and_matching_cover_material():
    source = Source(
        uuid='00000000-0000-0000-0000-000000000002',
        key='source',
        metadata={'identity_schema': 'source-content-v3'},
    )
    pages = [
        SourcePage(
            index=0,
            blocks=[
                SourceBlock(
                    uuid='block', block_type=BlockType.MARKDOWN, content='one'
                )
            ],
        )
    ]
    candidate = finalize_snapshot_identities(source, pages, [], [], [])
    persisted = candidate.model_copy(
        update={'source': candidate.source.model_copy(update={'revision': 1})}
    )
    assert_matching_snapshot(candidate, persisted)
    changed = persisted.model_copy(
        update={
            'pages': [
                SourcePage(
                    index=0,
                    blocks=[
                        SourceBlock(
                            uuid='block',
                            block_type=BlockType.MARKDOWN,
                            content='changed',
                        )
                    ],
                )
            ]
        }
    )
    with pytest.raises(SourceSnapshotConflict):
        assert_matching_snapshot(candidate, changed)


def test_finalized_snapshot_remaps_ordered_material_references():
    source = Source(uuid='00000000-0000-0000-0000-000000000004', key='source')
    pages = [
        SourcePage(
            index=0,
            blocks=[
                SourceBlock(
                    uuid='block-a', block_type=BlockType.MARKDOWN, content='A'
                ),
                SourceBlock(
                    uuid='block-b', block_type=BlockType.MARKDOWN, content='B'
                ),
            ],
        )
    ]
    instructions = [
        Instruction(
            uuid='instruction-old',
            member_block_uuids=['block-a'],
            governed_statement_uuids=['statement-old'],
        )
    ]
    statements = [
        StatementDraft(
            uuid='statement-old',
            content='Statement',
            embedding=[1.0, 2.0],
            member_block_uuids=['block-a', 'block-b'],
            is_exercise=True,
        )
    ]
    procedures = [
        ProcedureDraft(
            uuid='procedure-old',
            content='Procedure',
            embedding=[2.0, 3.0],
            member_block_uuids=['block-b'],
        )
    ]

    finalized = finalize_snapshot_identities(
        source, pages, instructions, statements, procedures
    )
    instruction = finalized.instructions[0]
    statement = finalized.statements[0]
    procedure = finalized.procedures[0]
    assert instruction.uuid != 'instruction-old'
    assert statement.uuid != 'statement-old'
    assert procedure.uuid != 'procedure-old'
    assert instruction.member_block_uuids == [finalized.pages[0].blocks[0].uuid]
    assert instruction.governed_statement_uuids == [statement.uuid]
    assert statement.member_block_uuids == [
        block.uuid for block in finalized.pages[0].blocks
    ]
    assert procedure.member_block_uuids == [finalized.pages[0].blocks[1].uuid]
    assert (
        finalize_snapshot_identities(
            source, pages, instructions, statements, procedures
        )
        == finalized
    )


def test_processed_fingerprint_rejects_unfinalized_media_digests():
    source = Source(uuid='00000000-0000-0000-0000-000000000003', key='source')
    page = SourcePage(
        index=0,
        blocks=[
            SourceBlock(
                block_type=BlockType.IMAGE,
                content=None,
                image_path='image.png',
                image_page_indexes=[0],
            )
        ],
    )
    with pytest.raises(
        ValueError, match='Source media digests must be finalized'
    ):
        processed_snapshot_fingerprint(_snapshot(source, [page]))


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


def test_source_embedding_profile_checks_text_and_images():
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
                        image_path='/image.png',
                        image_page_indexes=[0],
                        embedding=[1.0, 2.0],
                    ),
                ],
            )
        ],
    )
    assert_source_embedding_profile(snapshot, model='model-a', dimension=2)
    with pytest.raises(ValueError, match='Source embedding profile mismatch'):
        assert_source_embedding_profile(snapshot, model='model-b', dimension=2)


def test_source_schema_guard_rejects_old_records():
    require_source_schema('source-content-v3')
    with pytest.raises(
        ValueError, match='Source format is obsolete; re-ingest the source'
    ):
        require_source_schema(None)
