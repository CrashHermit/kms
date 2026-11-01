"""Persistence access for complete source structural graphs."""

from collections.abc import Callable

from neo4j import AsyncSession, AsyncTransaction

from kms2.core.model.page import SourcePage
from kms2.core.model.source import Source
from kms2.core.model.source_pedagogical.instruction import Instruction
from kms2.core.model.source_pedagogical.pedagogical import (
    ProcedureDraft,
    StatementDraft,
)
from kms2.database.source.queries.source_graph import (
    LOCK_EXISTING_SOURCE_WRITE,
    LOCK_SOURCE_WRITE,
    READ_SOURCE_CAPTURE_PAGES,
    REPLACE_SOURCE,
    SOURCE_EVIDENCE_REFERENCED,
    SOURCE_HAS_CAPTURE_DIGEST,
)
from kms2.database.source.queries.source_pedagogical import (
    LOAD_SOURCE_PAGES,
    REPLACE_SOURCE_PEDAGOGICAL_MATERIAL,
)
from kms2.database.source.source_graph_records import (
    source_graph_parameters,
    source_pages_from_records,
)


class SourceGraphRepository:
    """Replace complete persisted source structural graphs."""

    def __init__(self, session_factory: Callable[[], AsyncSession]) -> None:
        self._session_factory = session_factory

    async def replace_source(
        self, source: Source, pages: list[SourcePage]
    ) -> Source:
        """Replace a complete unreferenced source graph in one transaction."""
        parameters = source_graph_parameters(source, pages)
        async with self._session_factory() as session:
            async with await session.begin_transaction() as transaction:
                await self._lock_source(
                    transaction, source.uuid, allow_create=True
                )
                if await self._source_is_referenced(transaction, source.uuid):
                    raise ValueError(
                        'Source evidence is referenced; ingest a new source snapshot instead'
                    )
                if await self._has_capture_digest(transaction, source.uuid):
                    raise ValueError(
                        'Source snapshot capture is immutable; ingest a new source snapshot instead'
                    )
                result = await transaction.run(REPLACE_SOURCE, **parameters)
                record = await result.single(strict=True)
        return source.model_copy(update={'revision': record['revision']})

    async def _lock_source(
        self,
        transaction: AsyncTransaction,
        source_uuid: str,
        *,
        allow_create: bool = False,
    ) -> None:
        query = (
            LOCK_SOURCE_WRITE if allow_create else LOCK_EXISTING_SOURCE_WRITE
        )
        result = await transaction.run(query, source_uuid=source_uuid)
        await result.single(strict=True)

    async def _source_is_referenced(
        self, transaction: AsyncTransaction, source_uuid: str
    ) -> bool:
        result = await transaction.run(
            SOURCE_EVIDENCE_REFERENCED, source_uuid=source_uuid
        )
        record = await result.single(strict=True)
        return record['referenced']

    async def _has_capture_digest(
        self, transaction: AsyncTransaction, source_uuid: str
    ) -> bool:
        result = await transaction.run(
            SOURCE_HAS_CAPTURE_DIGEST, source_uuid=source_uuid
        )
        record = await result.single(strict=True)
        return record['captured']

    async def load_pages(self, source_uuid: str) -> list[SourcePage]:
        """Load persisted pages and their canonical ordered block data."""
        async with self._session_factory() as session:
            result = await session.run(
                LOAD_SOURCE_PAGES, source_uuid=source_uuid
            )
            record = await result.single(strict=True)
        return source_pages_from_records(record['pages'])

    async def replace_pedagogical_material(
        self,
        source_uuid: str,
        pages: list[SourcePage],
        instructions: list[Instruction],
        statements: list[StatementDraft],
        procedures: list[ProcedureDraft],
    ) -> None:
        """Replace pedagogical annotations and split-block topology."""
        blocks = [block for page in pages for block in page.blocks]
        parameters = {
            'source_uuid': source_uuid,
            'pages': [
                {
                    'index': page.index,
                    'markdown': page.markdown,
                    'block_uuids': [block.uuid for block in page.blocks],
                }
                for page in pages
            ],
            'blocks': [
                {
                    'uuid': block.uuid,
                    'block_type': block.block_type.value,
                    'content': block.content,
                    'embedding': block.embedding,
                    'crop_path': block.crop_path,
                    'crop_bbox': (
                        list(block.crop_bbox)
                        if block.crop_bbox is not None
                        else None
                    ),
                }
                for block in blocks
            ],
            'block_pairs': [
                {'from_uuid': block.uuid, 'to_uuid': next_block.uuid}
                for block, next_block in zip(blocks, blocks[1:], strict=False)
            ],
            'page_bounds': [
                {
                    'page_index': page.index,
                    'first_block_uuid': page.blocks[0].uuid,
                    'last_block_uuid': page.blocks[-1].uuid,
                }
                for page in pages
                if page.blocks
            ],
            'first_block_uuid': blocks[0].uuid if blocks else None,
            'last_block_uuid': blocks[-1].uuid if blocks else None,
            'instructions': [{'uuid': item.uuid} for item in instructions],
            'instruction_member_pairs': [
                {'block_uuid': block_uuid, 'instruction_uuid': item.uuid}
                for item in instructions
                for block_uuid in item.member_block_uuids
            ],
            'statements': [
                {
                    'uuid': item.uuid,
                    'is_exercise': item.is_exercise,
                    'content': item.content,
                    'embedding': item.embedding,
                }
                for item in statements
            ],
            'statement_member_pairs': [
                {'block_uuid': block_uuid, 'statement_uuid': item.uuid}
                for item in statements
                for block_uuid in item.member_block_uuids
            ],
            'procedures': [
                {
                    'uuid': item.uuid,
                    'content': item.content,
                    'embedding': item.embedding,
                }
                for item in procedures
            ],
            'procedure_member_pairs': [
                {'block_uuid': block_uuid, 'procedure_uuid': item.uuid}
                for item in procedures
                for block_uuid in item.member_block_uuids
            ],
            'instruction_governance_pairs': [
                {
                    'instruction_uuid': item.uuid,
                    'statement_uuid': statement_uuid,
                }
                for item in instructions
                for statement_uuid in item.governed_statement_uuids
            ],
        }
        async with self._session_factory() as session:
            async with await session.begin_transaction() as transaction:
                await self._lock_source(transaction, source_uuid)
                if await self._source_is_referenced(transaction, source_uuid):
                    raise ValueError(
                        'Source evidence is referenced; ingest a new source snapshot instead'
                    )
                if not await self._raw_pages_match(
                    transaction, source_uuid, pages
                ):
                    raise ValueError(
                        'Source snapshot capture is immutable; ingest a new source snapshot instead'
                    )
                result = await transaction.run(
                    REPLACE_SOURCE_PEDAGOGICAL_MATERIAL, **parameters
                )
                await result.single(strict=True)

    async def _raw_pages_match(
        self,
        transaction: AsyncTransaction,
        source_uuid: str,
        pages: list[SourcePage],
    ) -> bool:
        result = await transaction.run(
            READ_SOURCE_CAPTURE_PAGES, source_uuid=source_uuid
        )
        record = await result.single(strict=True)
        persisted = sorted(
            (
                (page['index'], page['markdown'])
                for page in record['pages']
                if page is not None
            ),
        )
        incoming = sorted((page.index, page.markdown) for page in pages)
        return persisted == incoming
