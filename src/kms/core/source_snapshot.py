"""Deterministic identities and conflict checks for processed source snapshots."""

from __future__ import annotations

import hashlib
import json
import math
from uuid import UUID, uuid5

from kms.core.model.block import SourceBlock
from kms.core.model.page import SourcePage
from kms.core.model.source import Source
from kms.core.model.source_processing.admission import SourceSnapshot

SOURCE_CONTENT_SCHEMA = 'source-content-v2'


class SourceSnapshotConflict(RuntimeError):
    """Processed content conflicts with the existing owned source snapshot."""


def processed_snapshot_fingerprint(pages: list[SourcePage]) -> str:
    """Hash ordered processed content and finalized media digests."""
    representation = {
        'schema': SOURCE_CONTENT_SCHEMA,
        'pages': [
            {
                'index': page.index,
                'markdown': page.markdown,
                'blocks': [
                    {
                        'block_type': block.block_type.value,
                        'content': block.content,
                        'crop_bbox': block.crop_bbox,
                        'crop_sha256': _crop_digest(block),
                        'assets': [
                            _asset_digest(asset.content_sha256)
                            for asset in block.assets
                        ],
                    }
                    for block in page.blocks
                ],
            }
            for page in pages
        ],
    }
    encoded = json.dumps(
        representation,
        sort_keys=True,
        ensure_ascii=False,
        separators=(',', ':'),
        allow_nan=False,
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def source_topology_fingerprint(pages: list[SourcePage]) -> str:
    """Hash current page order and block lineage separately from capture data."""
    representation = {
        'schema': 'source-topology-v1',
        'pages': [
            {
                'index': page.index,
                'markdown': page.markdown,
                'blocks': [
                    {
                        'uuid': block.uuid,
                        'origin_uuid': block.origin_uuid,
                        'block_type': block.block_type.value,
                        'content': block.content,
                        'crop_sha256': block.crop_sha256,
                        'crop_bbox': block.crop_bbox,
                        'assets': [
                            {
                                'uuid': asset.uuid,
                                'content_sha256': asset.content_sha256,
                            }
                            for asset in block.assets
                        ],
                    }
                    for block in page.blocks
                ],
            }
            for page in pages
        ],
    }
    encoded = json.dumps(
        representation,
        sort_keys=True,
        separators=(',', ':'),
        allow_nan=False,
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _crop_digest(block: SourceBlock) -> str | None:
    if block.crop_path is not None and block.crop_sha256 is None:
        raise ValueError(
            'Source media digests must be finalized before fingerprinting'
        )
    return block.crop_sha256


def _asset_digest(digest: str | None) -> str:
    if digest is None:
        raise ValueError(
            'Source media digests must be finalized before fingerprinting'
        )
    return digest


def assert_source_embedding_profile(
    snapshot: SourceSnapshot,
    *,
    model: str,
    dimension: int,
) -> None:
    """Require an exact embedding profile and finite vectors for indexed text."""
    expected_profile = {'model': model, 'dimension': dimension}
    if snapshot.source.metadata.get('embedding_profile') != expected_profile:
        raise ValueError('Source embedding profile mismatch; reindex required')
    for page in snapshot.pages:
        for block in page.blocks:
            if block.content and block.content.strip():
                vector = block.embedding
                if (
                    vector is None
                    or len(vector) != dimension
                    or any(
                        type(value) is not float or not math.isfinite(value)
                        for value in vector
                    )
                ):
                    raise ValueError(
                        'Source embedding profile mismatch; reindex required'
                    )


def finalize_snapshot_identities(
    source: Source, pages: list[SourcePage]
) -> SourceSnapshot:
    """Return an immutable processed snapshot with source-derived child IDs."""
    digest = processed_snapshot_fingerprint(pages)
    metadata = {
        **source.metadata,
        'processed_digest': digest,
        'identity_schema': SOURCE_CONTENT_SCHEMA,
    }
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
    return SourceSnapshot(
        source=source.model_copy(update={'metadata': metadata}),
        pages=finalized_pages,
    )


def assert_matching_snapshot(
    candidate: SourceSnapshot,
    persisted: SourceSnapshot,
) -> None:
    """Require captured content and stored derived topology to remain valid."""
    candidate_content_digest = processed_snapshot_fingerprint(candidate.pages)
    candidate_digest = candidate.source.metadata.get(
        'processed_digest', candidate_content_digest
    )
    persisted_topology_digest = persisted.source.metadata.get(
        'pedagogical_topology_digest'
    )
    persisted_digest = persisted.source.metadata.get('processed_digest')
    if persisted_topology_digest is None:
        persisted_content_digest = processed_snapshot_fingerprint(
            persisted.pages
        )
        topology_is_valid = (
            persisted_digest is None
            or persisted_digest == persisted_content_digest
        )
        persisted_digest = persisted_content_digest
    else:
        topology_is_valid = (
            persisted_topology_digest
            == source_topology_fingerprint(persisted.pages)
            and persisted_digest is not None
        )
    if (
        candidate.source.uuid != persisted.source.uuid
        or candidate_digest != candidate_content_digest
        or candidate_digest != persisted_digest
        or not topology_is_valid
        or candidate.source.metadata.get('snapshot_digest')
        != persisted.source.metadata.get('snapshot_digest')
        or candidate.source.metadata.get('normalization_identity_digest')
        != persisted.source.metadata.get('normalization_identity_digest')
    ):
        raise SourceSnapshotConflict(
            'Source snapshot content conflicts with the persisted source'
        )
