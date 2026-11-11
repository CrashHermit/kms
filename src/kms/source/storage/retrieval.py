"""Persistence access for ordered source blocks and block search."""

import json
from collections.abc import Callable
from typing import Any

from neo4j import AsyncSession

from kms.source.models.material import SourceProcedure, SourceStatement
from kms.source.models.snapshot import require_source_schema
from kms.source.models.source import Source, SourceBlock
from kms.source.retrieval.models import (
    SourceEvidenceBlock,
    SourceEvidenceGroup,
    SourceImageReference,
    SourcePassage,
    SourceProcedurePassage,
    SourceProcedureSearchHit,
    SourceSearchHit,
    SourceStatementPassage,
    SourceStatementSearchHit,
)
from kms.source.storage.records import native_datetime
from kms.source.storage.retrieval_queries import (
    READ_OWNED_SOURCE_IMAGES,
    READ_OWNED_SOURCE_PROCEDURES,
    READ_OWNED_SOURCE_STATEMENTS,
    SEARCH_OWNED_SOURCE_BLOCKS,
    SEARCH_OWNED_SOURCE_BLOCKS_VECTOR,
    SEARCH_OWNED_SOURCE_PROCEDURES,
    SEARCH_OWNED_SOURCE_PROCEDURES_VECTOR,
    SEARCH_OWNED_SOURCE_STATEMENTS,
    SEARCH_OWNED_SOURCE_STATEMENTS_VECTOR,
    read_source_passages_query,
)


class SourceRetrievalRepository:
    """Access owned source blocks, pedagogical material, and search results."""

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
        """Search embedded multimodal blocks within the user's owned sources."""
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
                query_embedding=query_embedding,
                candidate_limit=candidate_limit,
                source_uuids=source_uuids,
            )
            rows = await result.data()
        return [
            SourceSearchHit(
                source=_source_from_record(row['source']),
                block=_source_block_from_record(row['block']),
                page_indexes=row['page_indexes'],
                similarity=row['similarity'],
            )
            for row in rows
        ]

    async def search_owned_statements(
        self,
        user_uuid: str,
        query_embedding: list[float],
        *,
        candidate_limit: int,
        source_uuids: list[str] | None = None,
        is_exercise: bool | None = None,
    ) -> list[SourceStatementSearchHit]:
        """Search deduplicated statement occurrences in owned sources."""
        if source_uuids == []:
            return []
        async with self._session_factory() as session:
            result = await session.run(
                (
                    SEARCH_OWNED_SOURCE_STATEMENTS_VECTOR
                    if source_uuids is None and is_exercise is None
                    else SEARCH_OWNED_SOURCE_STATEMENTS
                ),
                user_uuid=user_uuid,
                query_embedding=query_embedding,
                candidate_limit=candidate_limit,
                source_uuids=source_uuids,
                is_exercise=is_exercise,
            )
            rows = await result.data()
        return [
            SourceStatementSearchHit(
                source=_source_from_record(row['source']),
                statement=SourceStatement(**row['statement']),
                page_indexes=row['page_indexes'],
                similarity=row['similarity'],
            )
            for row in rows
        ]

    async def search_owned_procedures(
        self,
        user_uuid: str,
        query_embedding: list[float],
        *,
        candidate_limit: int,
        source_uuids: list[str] | None = None,
    ) -> list[SourceProcedureSearchHit]:
        """Search deduplicated procedure occurrences in owned sources."""
        if source_uuids == []:
            return []
        async with self._session_factory() as session:
            result = await session.run(
                (
                    SEARCH_OWNED_SOURCE_PROCEDURES_VECTOR
                    if source_uuids is None
                    else SEARCH_OWNED_SOURCE_PROCEDURES
                ),
                user_uuid=user_uuid,
                query_embedding=query_embedding,
                candidate_limit=candidate_limit,
                source_uuids=source_uuids,
            )
            rows = await result.data()
        return [
            SourceProcedureSearchHit(
                source=_source_from_record(row['source']),
                procedure=SourceProcedure(**row['procedure']),
                page_indexes=row['page_indexes'],
                similarity=row['similarity'],
            )
            for row in rows
        ]

    async def read_statements(
        self,
        user_uuid: str,
        statement_uuids: list[str],
    ) -> list[SourceStatementPassage]:
        """Read owned statements and their full member/governing passages."""
        requested = list(dict.fromkeys(statement_uuids))
        if not requested:
            return []
        rows = []
        async with self._session_factory() as session:
            for statement_uuid in requested:
                result = await session.run(
                    READ_OWNED_SOURCE_STATEMENTS,
                    user_uuid=user_uuid,
                    statement_uuids=[statement_uuid],
                )
                rows.append(await result.single(strict=True))
        for row in rows:
            require_source_schema(row['source'].get('identity_schema'))
        statements_by_uuid = {
            row['statement']['uuid']: (
                SourceStatement(**row['statement']),
                _source_from_record(row['source']),
            )
            for row in rows
        }

        records: list[SourceStatementPassage] = []
        for statement_uuid in requested:
            statement, source = statements_by_uuid[statement_uuid]
            passages = await self.read_passages(
                user_uuid,
                statement.member_block_uuids,
            )
            records.append(
                SourceStatementPassage(
                    statement=statement,
                    passages=[
                        passage
                        for passage in passages
                        if passage.source.uuid == source.uuid
                    ],
                )
            )
        return records

    async def read_procedures(
        self,
        user_uuid: str,
        procedure_uuids: list[str],
    ) -> list[SourceProcedurePassage]:
        """Read owned procedures and their full member passages."""
        requested = list(dict.fromkeys(procedure_uuids))
        if not requested:
            return []
        rows = []
        async with self._session_factory() as session:
            for procedure_uuid in requested:
                result = await session.run(
                    READ_OWNED_SOURCE_PROCEDURES,
                    user_uuid=user_uuid,
                    procedure_uuids=[procedure_uuid],
                )
                rows.append(await result.single(strict=True))
        for row in rows:
            require_source_schema(row['source'].get('identity_schema'))
        procedures_by_uuid = {
            row['procedure']['uuid']: (
                SourceProcedure(**row['procedure']),
                _source_from_record(row['source']),
            )
            for row in rows
        }

        records: list[SourceProcedurePassage] = []
        for procedure_uuid in requested:
            procedure, source = procedures_by_uuid[procedure_uuid]
            passages = await self.read_passages(
                user_uuid,
                procedure.member_block_uuids,
            )
            records.append(
                SourceProcedurePassage(
                    procedure=procedure,
                    passages=[
                        passage
                        for passage in passages
                        if passage.source.uuid == source.uuid
                    ],
                )
            )
        return records

    async def read_passages(
        self,
        user_uuid: str,
        block_uuids: list[str],
        *,
        expand_memberships: bool = True,
        before: int = 0,
        after: int = 0,
    ) -> list[SourcePassage]:
        """Read owned blocks with selected group boundaries and neighbors."""
        if not block_uuids:
            return []

        async with self._session_factory() as session:
            result = await session.run(
                read_source_passages_query(before=before, after=after),
                user_uuid=user_uuid,
                block_uuids=block_uuids,
                expand_memberships=expand_memberships,
            )
            rows = await result.data()
        passages: list[SourcePassage] = []
        for row in rows:
            require_source_schema(row['source'].get('identity_schema'))
            evidence_blocks = [
                SourceEvidenceBlock(
                    block=_source_block_from_record(item['block']),
                    page_indexes=item['page_indexes'],
                )
                for item in row['blocks']
            ]
            block_positions = {
                item.block.uuid: index
                for index, item in enumerate(evidence_blocks)
            }
            groups = [
                SourceEvidenceGroup(
                    uuid=group['uuid'],
                    kind=group['kind'],
                    member_block_uuids=sorted(
                        group['member_block_uuids'],
                        key=block_positions.__getitem__,
                    ),
                    governed_statement_uuids=sorted(
                        group['governed_statement_uuids']
                    ),
                )
                for group in row['groups']
            ]
            passages.append(
                SourcePassage(
                    source=_source_from_record(row['source']),
                    selected_block_uuids=sorted(
                        row['selected_block_uuids'],
                        key=block_positions.__getitem__,
                    ),
                    blocks=evidence_blocks,
                    groups=sorted(
                        groups, key=lambda group: (group.kind, group.uuid)
                    ),
                )
            )
        passages.sort(
            key=lambda passage: (passage.source.key, passage.source.uuid)
        )
        return passages

    async def load_images(
        self, user_uuid: str, block_uuids: list[str]
    ) -> list[SourceImageReference]:
        """Resolve image paths and digests through requested owned blocks."""
        requested = list(dict.fromkeys(block_uuids))
        if not requested:
            return []
        async with self._session_factory() as session:
            result = await session.run(
                READ_OWNED_SOURCE_IMAGES,
                user_uuid=user_uuid,
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
    """Hydrate a source catalog summary from a database projection."""
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
    """Hydrate selected block data without its stored embedding."""
    return SourceBlock(
        uuid=record['uuid'],
        created_at=native_datetime(record['created_at']),
        updated_at=native_datetime(record['updated_at']),
        invalidated_at=native_datetime(record['invalidated_at'])
        if record['invalidated_at'] is not None
        else None,
        block_type=record['block_type'],
        content=record['content'],
        origin_uuid=record.get('origin_uuid'),
        crop_path=record['crop_path'],
        crop_sha256=record['crop_sha256'],
        crop_bbox=(
            tuple(record['crop_bbox'])
            if record['crop_bbox'] is not None
            else None
        ),
        image_path=record.get('image_path'),
        image_sha256=record.get('image_sha256'),
        image_page_indexes=record.get('image_page_indexes', []),
    )
