"""Deterministic identities and conflict checks for processed source snapshots."""

from __future__ import annotations

import hashlib
import json
import math
from uuid import UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field

from kms.source.models.material import (
    Instruction,
    ProcedureDraft,
    SourceProcedure,
    SourceStatement,
    StatementDraft,
)
from kms.source.models.source import Source, SourcePage


class SourceSnapshot(BaseModel):
    """Complete source content and retained pedagogical material."""

    model_config = ConfigDict(extra='forbid')

    source: Source
    pages: list[SourcePage]
    instructions: list[Instruction] = Field(default_factory=list)
    statements: list[SourceStatement] = Field(default_factory=list)
    procedures: list[SourceProcedure] = Field(default_factory=list)


SOURCE_CONTENT_SCHEMA = 'source-content-v3'


class SourceSnapshotConflict(RuntimeError):
    """Processed content conflicts with the existing owned source snapshot."""


def require_source_schema(identity_schema: str | None) -> None:
    """Reject source records written before the direct-image snapshot format."""
    if identity_schema != SOURCE_CONTENT_SCHEMA:
        raise ValueError('Source format is obsolete; re-ingest the source')


def processed_snapshot_fingerprint(snapshot: SourceSnapshot) -> str:
    """Hash ordered pages, image provenance, and complete pedagogical material."""
    block_positions = {
        block.uuid: [page_ordinal, block_ordinal]
        for page_ordinal, page in enumerate(snapshot.pages)
        for block_ordinal, block in enumerate(page.blocks)
    }
    statement_ordinals = {
        statement.uuid: ordinal
        for ordinal, statement in enumerate(snapshot.statements)
    }

    def positions(uuids: list[str]) -> list[list[int]]:
        try:
            return [block_positions[value] for value in uuids]
        except KeyError as error:
            raise ValueError(
                'Snapshot material references an unknown block'
            ) from error

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
                        'crop_sha256': _media_digest(
                            block.crop_path, block.crop_sha256
                        ),
                        'image_sha256': _media_digest(
                            block.image_path, block.image_sha256
                        ),
                        'image_page_indexes': block.image_page_indexes,
                        'origin': (
                            block_positions[block.origin_uuid]
                            if block.origin_uuid in block_positions
                            else block.origin_uuid
                        ),
                    }
                    for block in page.blocks
                ],
            }
            for page in snapshot.pages
        ],
        'instructions': [
            {
                'members': positions(item.member_block_uuids),
                'governed_statements': [
                    statement_ordinals[value]
                    for value in item.governed_statement_uuids
                ],
            }
            for item in snapshot.instructions
        ],
        'statements': [
            {
                'content': item.content,
                'is_exercise': item.is_exercise,
                'members': positions(item.member_block_uuids),
            }
            for item in snapshot.statements
        ],
        'procedures': [
            {
                'content': item.content,
                'members': positions(item.member_block_uuids),
            }
            for item in snapshot.procedures
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
    """Require an exact embedding profile and finite vectors for indexed data."""
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

    for page in snapshot.pages:
        for block in page.blocks:
            if (
                block.content is not None and block.content.strip()
            ) or block.image_path is not None:
                require_vector(block.embedding)
    for item in [*snapshot.statements, *snapshot.procedures]:
        require_vector(item.embedding)


def finalize_snapshot_identities(
    source: Source,
    pages: list[SourcePage],
    instructions: list[Instruction],
    statements: list[StatementDraft],
    procedures: list[ProcedureDraft],
) -> SourceSnapshot:
    """Assign stable IDs and remap memberships and instruction governance."""
    block_positions = {
        block.uuid: [page_ordinal, block_ordinal]
        for page_ordinal, page in enumerate(pages)
        for block_ordinal, block in enumerate(page.blocks)
    }
    statement_ordinals = {
        item.uuid: ordinal for ordinal, item in enumerate(statements)
    }

    def positions(uuids: list[str]) -> list[list[int]]:
        try:
            return [block_positions[value] for value in uuids]
        except KeyError as error:
            raise ValueError(
                'Snapshot material references an unknown block'
            ) from error

    # The digest is based on source positions/content, never provisional UUIDs.
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
                        'crop_sha256': _media_digest(
                            block.crop_path, block.crop_sha256
                        ),
                        'image_sha256': _media_digest(
                            block.image_path, block.image_sha256
                        ),
                        'image_page_indexes': block.image_page_indexes,
                        'origin': block_positions.get(
                            block.origin_uuid, block.origin_uuid
                        ),
                    }
                    for block in page.blocks
                ],
            }
            for page in pages
        ],
        'instructions': [
            {
                'members': positions(item.member_block_uuids),
                'governed_statements': [
                    statement_ordinals[value]
                    for value in item.governed_statement_uuids
                ],
            }
            for item in instructions
        ],
        'statements': [
            {
                'content': item.content,
                'is_exercise': item.is_exercise,
                'members': positions(item.member_block_uuids),
            }
            for item in statements
        ],
        'procedures': [
            {
                'content': item.content,
                'members': positions(item.member_block_uuids),
            }
            for item in procedures
        ],
    }
    digest = hashlib.sha256(
        json.dumps(
            representation,
            sort_keys=True,
            ensure_ascii=False,
            separators=(',', ':'),
            allow_nan=False,
        ).encode('utf-8')
    ).hexdigest()
    block_map = {
        block.uuid: str(
            uuid5(
                UUID(source.uuid),
                f'processed:{digest}:block:{page_ordinal}:{block_ordinal}',
            )
        )
        for page_ordinal, page in enumerate(pages)
        for block_ordinal, block in enumerate(page.blocks)
    }
    statement_map = {
        item.uuid: str(
            uuid5(UUID(source.uuid), f'processed:{digest}:statement:{ordinal}')
        )
        for ordinal, item in enumerate(statements)
    }

    def remap_blocks(uuids: list[str]) -> list[str]:
        try:
            return [block_map[value] for value in uuids]
        except KeyError as error:
            raise ValueError(
                'Snapshot material references an unknown block'
            ) from error

    finalized_pages = [
        page.model_copy(
            update={
                'blocks': [
                    block.model_copy(
                        update={
                            'uuid': block_map[block.uuid],
                            'origin_uuid': block_map.get(
                                block.origin_uuid, block.origin_uuid
                            ),
                        }
                    )
                    for block in page.blocks
                ]
            }
        )
        for page in pages
    ]
    finalized_instructions = [
        item.model_copy(
            update={
                'uuid': str(
                    uuid5(
                        UUID(source.uuid),
                        f'processed:{digest}:instruction:{ordinal}',
                    )
                ),
                'member_block_uuids': remap_blocks(item.member_block_uuids),
                'governed_statement_uuids': [
                    statement_map[value]
                    for value in item.governed_statement_uuids
                ],
            }
        )
        for ordinal, item in enumerate(instructions)
    ]
    finalized_statements = [
        SourceStatement(
            uuid=statement_map[item.uuid],
            content=item.content,
            is_exercise=item.is_exercise,
            embedding=item.embedding,
            member_block_uuids=remap_blocks(item.member_block_uuids),
        )
        for item in statements
    ]
    finalized_procedures = [
        SourceProcedure(
            uuid=str(
                uuid5(
                    UUID(source.uuid), f'processed:{digest}:procedure:{ordinal}'
                )
            ),
            content=item.content,
            embedding=item.embedding,
            member_block_uuids=remap_blocks(item.member_block_uuids),
        )
        for ordinal, item in enumerate(procedures)
    ]
    return SourceSnapshot(
        source=source.model_copy(
            update={
                'metadata': {
                    **source.metadata,
                    'processed_digest': digest,
                    'identity_schema': SOURCE_CONTENT_SCHEMA,
                }
            }
        ),
        pages=finalized_pages,
        instructions=finalized_instructions,
        statements=finalized_statements,
        procedures=finalized_procedures,
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
