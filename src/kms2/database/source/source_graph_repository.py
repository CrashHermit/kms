"""Persistence access for complete source structural graphs."""

import json
from collections.abc import Callable

from kms2.core.model import (
    Instruction,
    ProcedureDraft,
    Source,
    SourceBlock,
    SourcePage,
    StatementDraft,
)
from kms2.database.source.queries.source_graph import REPLACE_SOURCE


class SourceGraphRepository:
    """Replace complete persisted source structural graphs."""

    def __init__(self, session_factory: Callable) -> None:
        self._session_factory = session_factory

    async def replace_source(
        self,
        source: Source,
        pages: list[SourcePage],
        instructions: list[Instruction],
        statements: list[StatementDraft],
        procedures: list[ProcedureDraft],
    ) -> None:
        """Replace the persisted structural graph for one source."""
        blocks: list[SourceBlock] = [
            block for page in pages for block in page.blocks
        ]
        block_rows = [
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
        ]
        assets = [asset for block in blocks for asset in block.assets]
        assets_by_block = [
            (block, asset) for block in blocks for asset in block.assets
        ]
        asset_pairs = [
            {
                'from_uuid': asset.uuid,
                'to_uuid': next_asset.uuid,
            }
            for block in blocks
            for asset, next_asset in zip(
                block.assets, block.assets[1:], strict=False
            )
        ]
        page_blocks = [(page, block) for page in pages for block in page.blocks]
        page_block_pairs = [
            {'page_index': page.index, 'block_uuid': block.uuid}
            for page, block in page_blocks
        ]
        page_bounds = [
            {
                'page_index': page.index,
                'first_block_uuid': page.blocks[0].uuid,
                'last_block_uuid': page.blocks[-1].uuid,
            }
            for page in pages
            if page.blocks
        ]

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
            'blocks': block_rows,
            'assets': [
                {'uuid': asset.uuid, 'path': asset.path} for asset in assets
            ],
            'block_asset_pairs': [
                {'block_uuid': block.uuid, 'asset_uuid': asset.uuid}
                for block, asset in assets_by_block
            ],
            'asset_pairs': asset_pairs,
            'asset_bounds': [
                {
                    'block_uuid': block.uuid,
                    'first_asset_uuid': block.assets[0].uuid,
                    'last_asset_uuid': block.assets[-1].uuid,
                }
                for block in blocks
                if block.assets
            ],
            'page_block_pairs': page_block_pairs,
            'page_pairs': [
                {'from_index': page.index, 'to_index': next_page.index}
                for page, next_page in zip(pages, pages[1:], strict=False)
            ],
            'block_pairs': [
                {'from_uuid': block.uuid, 'to_uuid': next_block.uuid}
                for block, next_block in zip(blocks, blocks[1:], strict=False)
            ],
            'page_bounds': page_bounds,
            'first_page_index': pages[0].index if pages else None,
            'first_block_uuid': blocks[0].uuid if blocks else None,
            'instructions': [
                {'uuid': instruction.uuid} for instruction in instructions
            ],
            'instruction_member_pairs': [
                {
                    'block_uuid': block_uuid,
                    'instruction_uuid': instruction.uuid,
                }
                for instruction in instructions
                for block_uuid in instruction.member_block_uuids
            ],
            'statements': [
                {
                    'uuid': statement.uuid,
                    'source_uuid': source.uuid,
                    'is_exercise': statement.is_exercise,
                }
                for statement in statements
            ],
            'procedures': [
                {'uuid': procedure.uuid, 'source_uuid': source.uuid}
                for procedure in procedures
            ],
            'statement_member_pairs': [
                {
                    'block_uuid': block_uuid,
                    'statement_uuid': statement.uuid,
                }
                for statement in statements
                for block_uuid in statement.member_block_uuids
            ],
            'procedure_member_pairs': [
                {
                    'block_uuid': block_uuid,
                    'procedure_uuid': procedure.uuid,
                }
                for procedure in procedures
                for block_uuid in procedure.member_block_uuids
            ],
            'instruction_governance_pairs': [
                {
                    'instruction_uuid': instruction.uuid,
                    'statement_uuid': statement_uuid,
                }
                for instruction in instructions
                for statement_uuid in instruction.governed_statement_uuids
            ],
            'last_block_uuid': blocks[-1].uuid if blocks else None,
        }
        async with self._session_factory() as session:
            result = await session.run(REPLACE_SOURCE, **parameters)
            await result.consume()
