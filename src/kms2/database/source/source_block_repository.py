"""Persistence access for ordered source blocks and block search."""

from collections.abc import Callable

from kms2.core.model import SourceBlock, SourceBlockSimilarityMatch
from kms2.database.source.queries.source_blocks import (
    FIND_SIMILAR_SOURCE_BLOCKS,
    READ_SOURCE_BLOCKS,
)


class SourceBlockRepository:
    """Access ordered persisted source blocks and similar blocks."""

    def __init__(self, session_factory: Callable) -> None:
        self._session_factory = session_factory

    async def load_blocks(self, source_uuid: str) -> list[SourceBlock]:
        """Load canonical source blocks in persisted stream order."""
        async with self._session_factory() as session:
            result = await session.run(
                READ_SOURCE_BLOCKS,
                source_uuid=source_uuid,
            )
            rows = await result.data()
        return [
            SourceBlock(
                uuid=row['uuid'],
                block_type=row['block_type'],
                content=row['content'],
            )
            for row in rows
        ]

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
