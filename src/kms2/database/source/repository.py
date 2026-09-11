"""Persistence repository for the final KMS2 source structure."""

import json
from collections.abc import Callable

from kms2.core.model import Source, SourceBlock, SourcePage

from .queries import REPLACE_SOURCE


class SourceRepository:
    """Persists a source and its ordered page, block, and asset structure."""

    def __init__(self, session_factory: Callable) -> None:
        self._session_factory = session_factory

    async def replace_source(
        self,
        source: Source,
        pages: list[SourcePage],
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
            'pages': [{'index': page.index} for page in pages],
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
            'last_block_uuid': blocks[-1].uuid if blocks else None,
        }
        async with self._session_factory() as session:
            result = await session.run(REPLACE_SOURCE, **parameters)
            await result.consume()
