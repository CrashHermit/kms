"""Deterministic identities and conflict checks for processed source snapshots."""

from __future__ import annotations

import hashlib
import json
import math
from uuid import UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field

from kms.source.models.source import Source
from kms.source.models.unit import SourceUnit


class SourceSnapshot(BaseModel):
    """Complete source metadata and its ordered units."""

    model_config = ConfigDict(extra='forbid')

    source: Source
    units: list[SourceUnit] = Field(default_factory=list)


SOURCE_CONTENT_SCHEMA = 'source-content-v5'


class SourceSnapshotConflict(RuntimeError):
    """Processed content conflicts with the existing owned source snapshot."""


def require_source_schema(identity_schema: str | None) -> None:
    """Reject source records written before the current snapshot format."""
    if identity_schema != SOURCE_CONTENT_SCHEMA:
        raise ValueError('Source format is obsolete; re-ingest the source')


def _canonical_fingerprint(units: list[SourceUnit]) -> str:
    representation = {
        'schema': SOURCE_CONTENT_SCHEMA,
        'units': [
            {
                'kind': unit.kind.value,
                'blocks': [
                    {
                        'block_type': block.block_type.value,
                        'content': block.content,
                        'crop_bbox': block.crop_bbox,
                        'crop_sha256': _media_digest(
                            block.crop_path, block.crop_sha256
                        ),
                        'image_sha256': _media_digest(
                            block.image_path, block.image_sha256
                        ),
                    }
                    for block in unit.blocks
                ],
            }
            for unit in units
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


def processed_snapshot_fingerprint(snapshot: SourceSnapshot) -> str:
    """Hash ordered unit content and finalized media digests."""
    return _canonical_fingerprint(snapshot.units)


def _media_digest(path: str | None, digest: str | None) -> str | None:
    if path is not None and digest is None:
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
    """Require the exact embedding profile and finite vectors for indexed data."""
    expected_profile = {'model': model, 'dimension': dimension}
    if snapshot.source.metadata.get('embedding_profile') != expected_profile:
        raise ValueError('Source embedding profile mismatch; reindex required')

    def require_vector(vector: list[float] | None) -> None:
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

    for unit in snapshot.units:
        if len(unit.blocks) > 1 or any(
            (block.content is not None and block.content.strip())
            or block.image_path is not None
            for block in unit.blocks
        ):
            require_vector(unit.embedding)
        for block in unit.blocks:
            if (
                block.content is not None and block.content.strip()
            ) or block.image_path is not None:
                require_vector(block.embedding)


def finalize_snapshot_identities(
    source: Source, units: list[SourceUnit]
) -> SourceSnapshot:
    """Assign deterministic unit/block IDs without changing their lifecycle."""
    digest = _canonical_fingerprint(units)
    finalized_units = [
        unit.model_copy(
            update={
                'uuid': str(
                    uuid5(
                        UUID(source.uuid),
                        f'processed:{digest}:unit:{unit_ordinal}',
                    )
                ),
                'blocks': [
                    block.model_copy(
                        update={
                            'uuid': str(
                                uuid5(
                                    UUID(source.uuid),
                                    f'processed:{digest}:unit:{unit_ordinal}:block:{block_ordinal}',
                                )
                            )
                        }
                    )
                    for block_ordinal, block in enumerate(unit.blocks)
                ],
            }
        )
        for unit_ordinal, unit in enumerate(units)
    ]
    return SourceSnapshot(
        source=source.model_copy(
            update={
                'metadata': {
                    **source.metadata,
                    'processed_digest': digest,
                    'identity_schema': SOURCE_CONTENT_SCHEMA,
                },
            }
        ),
        units=finalized_units,
    )


def assert_matching_snapshot(
    candidate: SourceSnapshot, persisted: SourceSnapshot
) -> None:
    """Require captured content and all persisted annotations to match."""
    candidate_digest = processed_snapshot_fingerprint(candidate)
    persisted_digest = processed_snapshot_fingerprint(persisted)
    if (
        candidate.source.uuid != persisted.source.uuid
        or candidate.source.metadata.get('processed_digest') != candidate_digest
        or persisted.source.metadata.get('processed_digest') != persisted_digest
        or candidate_digest != persisted_digest
        or candidate.source.metadata.get('snapshot_digest')
        != persisted.source.metadata.get('snapshot_digest')
    ):
        raise SourceSnapshotConflict(
            'Source snapshot content conflicts with the persisted source'
        )
