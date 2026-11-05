"""Persistence access for ordered source blocks and block search."""

import json
from collections.abc import Callable
from typing import Any

from neo4j import AsyncSession

from kms.core.model.block import SourceBlock
from kms.core.model.similarity import SourceBlockSimilarityMatch
from kms.core.model.source import Source
from kms.core.model.source_evidence import (
    SourceEvidenceBlock,
    SourceEvidenceGroup,
    SourcePassage,
    SourceProcedurePassage,
    SourceProcedureSearchCandidate,
    SourceSearchCandidate,
    SourceStatementPassage,
    SourceStatementSearchCandidate,
)
from kms.core.model.source_semantic.source_procedure import SourceProcedure
from kms.core.model.source_semantic.source_statement import SourceStatement
from kms.core.model.visual_asset import VisualAsset
from kms.database.source.queries.source_blocks import (
    FIND_SIMILAR_SOURCE_BLOCKS,
    READ_OWNED_SOURCE_ASSETS,
    READ_OWNED_SOURCE_ASSETS_AT_REVISIONS,
    READ_OWNED_SOURCE_PROCEDURES,
    READ_OWNED_SOURCE_STATEMENTS,
    READ_SOURCE_BLOCKS,
    SEARCH_OWNED_SOURCE_BLOCKS,
    SEARCH_OWNED_SOURCE_BLOCKS_VECTOR,
    SEARCH_OWNED_SOURCE_PROCEDURES,
    SEARCH_OWNED_SOURCE_PROCEDURES_VECTOR,
    SEARCH_OWNED_SOURCE_STATEMENTS,
    SEARCH_OWNED_SOURCE_STATEMENTS_VECTOR,
    read_source_passages_query,
)


class SourceBlockRepository:
    """Access ordered persisted source blocks and similar blocks."""

    def __init__(self, session_factory: Callable[[], AsyncSession]) -> None:
        self._session_factory = session_factory

    async def load_blocks(self, source_uuid: str) -> list[SourceBlock]:
        """Load canonical source blocks in persisted stream order."""
        async with self._session_factory() as session:
            result = await session.run(
                READ_SOURCE_BLOCKS,
                source_uuid=source_uuid,
            )
            rows = await result.data()
        return [_source_block_from_record(row) for row in rows]

    async def find_similar_blocks(
        self,
        block_uuid: str,
        *,
        top_k: int,
    ) -> list[SourceBlockSimilarityMatch]:
        """Find nearest source blocks using Neo4j's source-block index."""
        async with self._session_factory() as session:
            result = await session.run(
                FIND_SIMILAR_SOURCE_BLOCKS,
                query_uuid=block_uuid,
                top_k=top_k,
                candidate_limit=top_k + 1,
            )
            rows = await result.data()
        return [
            SourceBlockSimilarityMatch(
                uuid=row['uuid'],
                block_type=row['block_type'],
                content=row['content'],
                score=row['score'],
            )
            for row in rows
        ]

    async def search_owned_blocks(
        self,
        user_uuid: str,
        query_embedding: list[float],
        *,
        candidate_limit: int,
        source_uuids: list[str] | None = None,
    ) -> list[SourceSearchCandidate]:
        """Search embedded text blocks within the user's owned sources."""
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
            SourceSearchCandidate(
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
    ) -> list[SourceStatementSearchCandidate]:
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
            SourceStatementSearchCandidate(
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
    ) -> list[SourceProcedureSearchCandidate]:
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
            SourceProcedureSearchCandidate(
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

    async def load_assets(
        self,
        user_uuid: str,
        asset_uuids: list[str],
    ) -> list[VisualAsset]:
        """Load requested original assets attached to an owned source block."""
        requested_uuids = list(dict.fromkeys(asset_uuids))
        if not requested_uuids:
            return []

        async with self._session_factory() as session:
            result = await session.run(
                READ_OWNED_SOURCE_ASSETS,
                user_uuid=user_uuid,
                asset_uuids=requested_uuids,
            )
            rows = await result.data()
        assets_by_uuid = {
            row['uuid']: VisualAsset(
                uuid=row['uuid'],
                path=row['path'],
                content_sha256=row['content_sha256'],
            )
            for row in rows
        }
        return [
            assets_by_uuid[asset_uuid]
            for asset_uuid in requested_uuids
            if asset_uuid in assets_by_uuid
        ]

    async def load_assets_at_revisions(
        self,
        user_uuid: str,
        image_references: list[tuple[str, int, str]],
    ) -> list[VisualAsset]:
        """Load assets only when their owned source revision still matches."""
        if not image_references:
            return []
        async with self._session_factory() as session:
            result = await session.run(
                READ_OWNED_SOURCE_ASSETS_AT_REVISIONS,
                user_uuid=user_uuid,
                image_references=[
                    {
                        'position': position,
                        'source_uuid': source_uuid,
                        'revision': revision,
                        'asset_uuid': asset_uuid,
                    }
                    for position, (
                        source_uuid,
                        revision,
                        asset_uuid,
                    ) in enumerate(image_references)
                ],
            )
            rows = await result.data()
        return [
            VisualAsset(
                uuid=row['uuid'],
                path=row['path'],
                content_sha256=row['content_sha256'],
            )
            for row in rows
        ]


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
        metadata=metadata,
        revision=record.get('revision', 0),
    )


def _source_block_from_record(record: dict[str, Any]) -> SourceBlock:
    """Hydrate selected block data without its stored embedding."""
    return SourceBlock(
        uuid=record['uuid'],
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
        assets=[
            VisualAsset(
                uuid=asset['uuid'],
                path=asset['path'],
                content_sha256=asset['content_sha256'],
            )
            for asset in record['assets']
        ],
    )
