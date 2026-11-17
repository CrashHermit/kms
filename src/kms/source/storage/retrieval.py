"""Persistence access for owner-scoped source blocks and units."""

import json
from collections.abc import Callable
from typing import Any

from neo4j import AsyncSession

from kms.source.models.snapshot import (
    SOURCE_CONTENT_SCHEMA,
    require_source_schema,
)
from kms.source.models.source import Source, SourceBlock
from kms.source.models.unit import SourceUnit, SourceUnitKind
from kms.source.retrieval.models import (
    SourceBlockRead,
    SourceImageReference,
    SourceSearchHit,
    SourceUnitRead,
    SourceUnitSearchHit,
)
from kms.source.storage.records import native_datetime
from kms.source.storage.retrieval_queries import (
    READ_OWNED_SOURCE_BLOCKS,
    READ_OWNED_SOURCE_IMAGES,
    SEARCH_OWNED_SOURCE_BLOCKS,
    SEARCH_OWNED_SOURCE_BLOCKS_VECTOR,
    SEARCH_OWNED_SOURCE_UNITS,
    SEARCH_OWNED_SOURCE_UNITS_VECTOR,
    read_source_units_query,
)


class SourceRetrievalRepository:
    """Access owned source blocks, units, images, and search results."""

    def __init__(self, session_factory: Callable[[], AsyncSession]) -> None:
        self._session_factory = session_factory

    async def search_owned_blocks(
        self,
        user_uuid: str,
        query_embedding: list[float],
        *,
        candidate_limit: int,
        source_uuids: list[str] | None = None,
    ) -> list[SourceSearchHit]:
        """Search embedded blocks in current-schema owned sources."""
        if source_uuids == []:
            return []
        async with self._session_factory() as session:
            result = await session.run(
                (
                    SEARCH_OWNED_SOURCE_BLOCKS_VECTOR
                    if source_uuids is None
                    else SEARCH_OWNED_SOURCE_BLOCKS
                ),
                user_uuid=user_uuid,
                identity_schema=SOURCE_CONTENT_SCHEMA,
                query_embedding=query_embedding,
                candidate_limit=candidate_limit,
                source_uuids=source_uuids,
            )
            rows = await result.data()
        return [
            SourceSearchHit(
                source=_source_from_record(row['source']),
                unit_uuid=row['unit_uuid'],
                block=_source_block_from_record(row['block']),
                similarity=row['similarity'],
            )
            for row in rows
        ]

    async def search_owned_units(
        self,
        user_uuid: str,
        query_embedding: list[float],
        *,
        candidate_limit: int,
        source_uuids: list[str] | None = None,
        kind: SourceUnitKind | None = None,
    ) -> list[SourceUnitSearchHit]:
        """Search complete embedded units in current-schema owned sources."""
        if source_uuids == []:
            return []
        async with self._session_factory() as session:
            query = (
                SEARCH_OWNED_SOURCE_UNITS_VECTOR
                if source_uuids is None and kind is None
                else SEARCH_OWNED_SOURCE_UNITS
            )
            result = await session.run(
                query,
                user_uuid=user_uuid,
                identity_schema=SOURCE_CONTENT_SCHEMA,
                query_embedding=query_embedding,
                candidate_limit=candidate_limit,
                source_uuids=source_uuids,
                kind=kind.value if kind is not None else None,
            )
            rows = await result.data()
        return [
            SourceUnitSearchHit(
                source=_source_from_record(row['source']),
                unit=_source_unit_from_record(row['unit']),
                similarity=row['similarity'],
            )
            for row in rows
        ]

    async def read_blocks(
        self, user_uuid: str, block_uuids: list[str]
    ) -> list[SourceBlockRead]:
        """Read requested owned blocks in first-request order."""
        requested = list(dict.fromkeys(block_uuids))
        if not requested:
            return []
        async with self._session_factory() as session:
            result = await session.run(
                READ_OWNED_SOURCE_BLOCKS,
                user_uuid=user_uuid,
                identity_schema=SOURCE_CONTENT_SCHEMA,
                block_uuids=requested,
            )
            rows = await result.data()
        by_uuid = {
            row['block']['uuid']: SourceBlockRead(
                source=_source_from_record(row['source']),
                unit_uuid=row['unit_uuid'],
                block=_source_block_from_record(row['block']),
            )
            for row in rows
        }
        return [by_uuid[uuid] for uuid in requested if uuid in by_uuid]

    async def read_units(
        self,
        user_uuid: str,
        unit_uuids: list[str],
        *,
        before_units: int = 0,
        after_units: int = 0,
    ) -> list[SourceUnitRead]:
        """Read requested units and bounded same-source neighboring units."""
        requested = list(dict.fromkeys(unit_uuids))
        if not requested:
            return []
        query = read_source_units_query(before=before_units, after=after_units)
        rows = []
        async with self._session_factory() as session:
            for unit_uuid in requested:
                result = await session.run(
                    query,
                    user_uuid=user_uuid,
                    identity_schema=SOURCE_CONTENT_SCHEMA,
                    unit_uuids=[unit_uuid],
                )
                rows.extend(await result.data())
        sources: dict[
            str, tuple[Source, dict[str, tuple[int, SourceUnitRead]]]
        ] = {}
        for row in rows:
            source = _source_from_record(row['source'])
            require_source_schema(row['source'].get('identity_schema'))
            by_unit = sources.setdefault(source.uuid, (source, {}))[1]
            for unit_record in row['units']:
                unit = _source_unit_from_record(unit_record)
                by_unit[unit.uuid] = (
                    unit_record['order_index'],
                    SourceUnitRead(source=source, unit=unit),
                )
        return [
            read
            for _, units in sources.values()
            for _, read in sorted(units.values(), key=lambda value: value[0])
        ]

    async def load_images(
        self, user_uuid: str, block_uuids: list[str]
    ) -> list[SourceImageReference]:
        """Resolve image artifacts through requested owned source blocks."""
        requested = list(dict.fromkeys(block_uuids))
        if not requested:
            return []
        async with self._session_factory() as session:
            result = await session.run(
                READ_OWNED_SOURCE_IMAGES,
                user_uuid=user_uuid,
                identity_schema=SOURCE_CONTENT_SCHEMA,
                block_uuids=requested,
            )
            rows = await result.data()
        by_uuid: dict[str, SourceImageReference] = {}
        for row in rows:
            require_source_schema(row.get('identity_schema'))
            if row['block_type'] != 'image':
                continue
            if not row.get('image_path') or not row.get('image_sha256'):
                raise ValueError(
                    'Source image storage is incomplete or obsolete; re-ingest the source'
                )
            by_uuid[row['block_uuid']] = SourceImageReference(
                block_uuid=row['block_uuid'],
                image_path=row['image_path'],
                image_sha256=row['image_sha256'],
            )
        return [by_uuid[uuid] for uuid in requested if uuid in by_uuid]


def _source_from_record(record: dict[str, Any]) -> Source:
    """Hydrate a source summary from a database projection."""
    raw_metadata = record['metadata']
    metadata = (
        json.loads(raw_metadata)
        if isinstance(raw_metadata, str)
        else raw_metadata or {}
    )
    return Source(
        uuid=record['uuid'],
        key=record['key'],
        created_at=native_datetime(record['created_at']),
        updated_at=native_datetime(record['updated_at']),
        invalidated_at=native_datetime(record['invalidated_at'])
        if record['invalidated_at'] is not None
        else None,
        metadata=metadata,
        revision=record.get('revision', 0),
    )


def _source_block_from_record(record: dict[str, Any]) -> SourceBlock:
    """Hydrate a selected block without its stored embedding."""
    return SourceBlock(
        uuid=record['uuid'],
        created_at=native_datetime(record['created_at']),
        updated_at=native_datetime(record['updated_at']),
        invalidated_at=native_datetime(record['invalidated_at'])
        if record['invalidated_at'] is not None
        else None,
        block_type=record['block_type'],
        content=record['content'],
        crop_path=record.get('crop_path'),
        crop_sha256=record.get('crop_sha256'),
        crop_bbox=tuple(record['crop_bbox'])
        if record.get('crop_bbox') is not None
        else None,
        image_path=record.get('image_path'),
        image_sha256=record.get('image_sha256'),
    )


def _source_unit_from_record(record: dict[str, Any]) -> SourceUnit:
    """Hydrate a complete unit and its ordered nested blocks."""
    return SourceUnit(
        uuid=record['uuid'],
        created_at=native_datetime(record['created_at']),
        updated_at=native_datetime(record['updated_at']),
        invalidated_at=native_datetime(record['invalidated_at'])
        if record['invalidated_at'] is not None
        else None,
        kind=SourceUnitKind(record['kind']),
        blocks=[_source_block_from_record(block) for block in record['blocks']],
    )
