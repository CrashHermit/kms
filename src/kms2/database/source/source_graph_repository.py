"""Persistence access for complete source structural graphs."""

import json
from collections.abc import Callable

from neo4j import AsyncSession

from kms2.core.model.block import SourceBlock
from kms2.core.model.page import SourcePage
from kms2.core.model.source import Source
from kms2.core.model.source_pedagogical.instruction import Instruction
from kms2.core.model.source_pedagogical.pedagogical import (
    ProcedureDraft,
    StatementDraft,
)
from kms2.core.model.visual_asset import VisualAsset
from kms2.database.source.queries.source_graph import REPLACE_SOURCE
from kms2.database.source.queries.source_pedagogical import (
    LOAD_SOURCE_PAGES,
    REPLACE_SOURCE_PEDAGOGICAL_MATERIAL,
)


class SourceGraphRepository:
    """Replace complete persisted source structural graphs."""

    def __init__(self, session_factory: Callable[[], AsyncSession]) -> None:
        self._session_factory = session_factory

    async def replace_source(
        self, source: Source, pages: list[SourcePage]
    ) -> None:
        """Replace the persisted structural graph for one source."""
        blocks = [block for page in pages for block in page.blocks]
        assets = [asset for block in blocks for asset in block.assets]
        parameters = {
            'source': {
                'uuid': source.uuid,
                'key': source.key,
                'metadata': json.dumps(source.metadata, sort_keys=True),
            },
            'pages': [
                {'index': page.index, 'markdown': page.markdown}
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
            'assets': [
                {'uuid': asset.uuid, 'path': asset.path} for asset in assets
            ],
            'block_asset_pairs': [
                {'block_uuid': block.uuid, 'asset_uuid': asset.uuid}
                for block in blocks
                for asset in block.assets
            ],
            'asset_pairs': [
                {'from_uuid': asset.uuid, 'to_uuid': next_asset.uuid}
                for block in blocks
                for asset, next_asset in zip(
                    block.assets, block.assets[1:], strict=False
                )
            ],
            'asset_bounds': [
                {
                    'block_uuid': block.uuid,
                    'first_asset_uuid': block.assets[0].uuid,
                    'last_asset_uuid': block.assets[-1].uuid,
                }
                for block in blocks
                if block.assets
            ],
            'page_block_pairs': [
                {'page_index': page.index, 'block_uuid': block.uuid}
                for page in pages
                for block in page.blocks
            ],
            'page_pairs': [
                {'from_index': page.index, 'to_index': next_page.index}
                for page, next_page in zip(pages, pages[1:], strict=False)
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
            'first_page_index': pages[0].index if pages else None,
            'first_block_uuid': blocks[0].uuid if blocks else None,
            'last_block_uuid': blocks[-1].uuid if blocks else None,
        }
        async with self._session_factory() as session:
            result = await session.run(REPLACE_SOURCE, **parameters)
            await result.consume()

    async def load_pages(self, source_uuid: str) -> list[SourcePage]:
        """Load persisted pages and their canonical ordered block data."""
        async with self._session_factory() as session:
            result = await session.run(
                LOAD_SOURCE_PAGES, source_uuid=source_uuid
            )
            record = await result.single(strict=True)
        records = record['pages']
        pages: list[SourcePage] = []
        for record in records:
            blocks = [
                SourceBlock(
                    uuid=block['uuid'],
                    block_type=block['block_type'],
                    content=block['content'],
                    embedding=block['embedding'],
                    crop_path=block['crop_path'],
                    crop_bbox=block['crop_bbox'],
                    assets=[
                        VisualAsset(uuid=asset['uuid'], path=asset['path'])
                        for asset in block['assets']
                    ],
                )
                for block in record['blocks']
            ]
            pages.append(
                SourcePage(
                    index=record['index'],
                    markdown=record['markdown'],
                    blocks=blocks,
                )
            )
        return pages

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
            result = await session.run(
                REPLACE_SOURCE_PEDAGOGICAL_MATERIAL, **parameters
            )
            await result.consume()
