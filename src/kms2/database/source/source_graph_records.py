"""Shared serialization and hydration for structural source graphs."""

import json

from kms2.core.model.block import SourceBlock
from kms2.core.model.page import SourcePage
from kms2.core.model.source import Source
from kms2.core.model.visual_asset import VisualAsset


def source_graph_parameters(
    source: Source, pages: list[SourcePage]
) -> dict[str, object]:
    """Serialize a source and ordered structural pages for Cypher writes."""
    blocks = [block for page in pages for block in page.blocks]
    assets = [asset for block in blocks for asset in block.assets]
    return {
        'source': {
            'uuid': source.uuid,
            'key': source.key,
            'metadata': json.dumps(source.metadata, sort_keys=True),
        },
        'pages': [
            {'index': page.index, 'markdown': page.markdown} for page in pages
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


def source_pages_from_records(
    records: list[dict[str, object]],
) -> list[SourcePage]:
    """Hydrate ordered source pages and nested blocks/assets from query rows."""
    return [
        SourcePage(
            index=page['index'],
            markdown=page['markdown'],
            blocks=[
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
                for block in page['blocks']
            ],
        )
        for page in records
    ]
