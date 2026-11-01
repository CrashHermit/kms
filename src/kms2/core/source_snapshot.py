"""Deterministic identities and conflict checks for processed source snapshots."""

from __future__ import annotations

import hashlib
import json
from uuid import UUID, uuid5

from kms2.core.model.page import SourcePage
from kms2.core.model.source import Source
from kms2.core.model.source_processing.admission import (
    OwnedSourceSnapshot,
    StagedSourceSnapshot,
)


class SourceSnapshotConflict(RuntimeError):
    """A processed snapshot no longer matches the source selected for publication."""


def processed_snapshot_fingerprint(pages: list[SourcePage]) -> str:
    """Hash the ordered processed representation, excluding generated identities."""
    representation = [
        {
            'index': page.index,
            'markdown': page.markdown,
            'blocks': [
                {
                    'block_type': block.block_type.value,
                    'content': block.content,
                    'embedding': block.embedding,
                    'crop_path': block.crop_path,
                    'crop_bbox': block.crop_bbox,
                    'assets': [asset.path for asset in block.assets],
                }
                for block in page.blocks
            ],
        }
        for page in pages
    ]
    encoded = json.dumps(
        representation,
        sort_keys=True,
        ensure_ascii=False,
        separators=(',', ':'),
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def finalize_snapshot_identities(
    source: Source, pages: list[SourcePage]
) -> StagedSourceSnapshot:
    """Return an immutable processed snapshot with source-derived child IDs."""
    digest = processed_snapshot_fingerprint(pages)
    metadata = {**source.metadata, 'processed_digest': digest}
    finalized_pages = []
    for page_ordinal, page in enumerate(pages):
        blocks = []
        for block_ordinal, block in enumerate(page.blocks):
            block_uuid = str(
                uuid5(
                    UUID(source.uuid),
                    f'processed:{digest}:block:{page_ordinal}:{block_ordinal}',
                )
            )
            assets = [
                asset.model_copy(
                    update={
                        'uuid': str(
                            uuid5(
                                UUID(source.uuid),
                                'processed:'
                                f'{digest}:asset:{page_ordinal}:'
                                f'{block_ordinal}:{asset_ordinal}',
                            )
                        )
                    }
                )
                for asset_ordinal, asset in enumerate(block.assets)
            ]
            blocks.append(
                block.model_copy(update={'uuid': block_uuid, 'assets': assets})
            )
        finalized_pages.append(page.model_copy(update={'blocks': blocks}))
    return StagedSourceSnapshot(
        source=source.model_copy(update={'metadata': metadata}),
        pages=finalized_pages,
    )


def assert_matching_snapshot(
    candidate: StagedSourceSnapshot,
    persisted: OwnedSourceSnapshot | StagedSourceSnapshot,
) -> None:
    """Require the same source revision, processed content, and child identities."""
    candidate_block_uuids = [
        block.uuid for page in candidate.pages for block in page.blocks
    ]
    persisted_block_uuids = [
        block.uuid for page in persisted.pages for block in page.blocks
    ]
    candidate_asset_uuids = [
        asset.uuid
        for page in candidate.pages
        for block in page.blocks
        for asset in block.assets
    ]
    persisted_asset_uuids = [
        asset.uuid
        for page in persisted.pages
        for block in page.blocks
        for asset in block.assets
    ]
    if (
        candidate.source.uuid != persisted.source.uuid
        or candidate.source.revision != persisted.source.revision
        or processed_snapshot_fingerprint(candidate.pages)
        != processed_snapshot_fingerprint(persisted.pages)
        or candidate_block_uuids != persisted_block_uuids
        or candidate_asset_uuids != persisted_asset_uuids
    ):
        raise SourceSnapshotConflict(
            'Processed source snapshot changed before publication'
        )
