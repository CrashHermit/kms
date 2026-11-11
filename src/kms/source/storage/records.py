"""Shared serialization and hydration for structural source graphs."""

import json
from datetime import datetime
from typing import Any

from kms.core.models import Edge
from kms.source.models.snapshot import SourceSnapshot
from kms.source.models.source import SourceBlock, SourcePage


def native_datetime(value: Any) -> datetime:
    """Convert a Neo4j temporal value at the database boundary."""
    return value if isinstance(value, datetime) else value.to_native()


def source_graph_parameters(snapshot: SourceSnapshot) -> dict[str, object]:
    """Serialize a complete source snapshot for one structural transaction."""
    source, pages = snapshot.source, snapshot.pages
    blocks = [block for page in pages for block in page.blocks]
    return {
        'edge_created_at': Edge().created_at,
        'source': {
            'uuid': source.uuid,
            'key': source.key,
            'metadata': json.dumps(source.metadata, sort_keys=True),
            'identity_schema': source.metadata.get('identity_schema'),
            'created_at': source.created_at,
            'updated_at': source.updated_at,
            'invalidated_at': source.invalidated_at,
        },
        'pages': [
            {'index': page.index, 'markdown': page.markdown} for page in pages
        ],
        'blocks': [
            {
                'uuid': block.uuid,
                'created_at': block.created_at,
                'updated_at': block.updated_at,
                'invalidated_at': block.invalidated_at,
                'block_type': block.block_type.value,
                'content': block.content,
                'origin_uuid': block.origin_uuid,
                'embedding': block.embedding,
                'crop_path': block.crop_path,
                'crop_sha256': block.crop_sha256,
                'crop_bbox': list(block.crop_bbox)
                if block.crop_bbox is not None
                else None,
                'image_path': block.image_path,
                'image_sha256': block.image_sha256,
                'image_page_indexes': block.image_page_indexes,
            }
            for block in blocks
        ],
        'page_block_pairs': [
            {'page_index': page.index, 'block_uuid': block.uuid}
            for page in pages
            for block in page.blocks
        ],
        'page_pairs': [
            {'from_index': page.index, 'to_index': nxt.index}
            for page, nxt in zip(pages, pages[1:], strict=False)
        ],
        'block_pairs': [
            {'from_uuid': block.uuid, 'to_uuid': nxt.uuid}
            for block, nxt in zip(blocks, blocks[1:], strict=False)
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
        'instructions': [
            {
                'uuid': item.uuid,
                'ordinal': ordinal,
                'created_at': item.created_at,
                'updated_at': item.updated_at,
                'invalidated_at': item.invalidated_at,
            }
            for ordinal, item in enumerate(snapshot.instructions)
        ],
        'statements': [
            {
                'uuid': item.uuid,
                'content': item.content,
                'is_exercise': item.is_exercise,
                'embedding': item.embedding,
                'ordinal': ordinal,
            }
            for ordinal, item in enumerate(snapshot.statements)
        ],
        'procedures': [
            {
                'uuid': item.uuid,
                'content': item.content,
                'embedding': item.embedding,
                'ordinal': ordinal,
            }
            for ordinal, item in enumerate(snapshot.procedures)
        ],
        'instruction_member_pairs': [
            {
                'instruction_uuid': item.uuid,
                'block_uuid': block_uuid,
                'position': position,
            }
            for item in snapshot.instructions
            for position, block_uuid in enumerate(item.member_block_uuids)
        ],
        'statement_member_pairs': [
            {
                'statement_uuid': item.uuid,
                'block_uuid': block_uuid,
                'position': position,
            }
            for item in snapshot.statements
            for position, block_uuid in enumerate(item.member_block_uuids)
        ],
        'procedure_member_pairs': [
            {
                'procedure_uuid': item.uuid,
                'block_uuid': block_uuid,
                'position': position,
            }
            for item in snapshot.procedures
            for position, block_uuid in enumerate(item.member_block_uuids)
        ],
        'instruction_governance_pairs': [
            {
                'instruction_uuid': item.uuid,
                'statement_uuid': statement_uuid,
                'position': position,
            }
            for item in snapshot.instructions
            for position, statement_uuid in enumerate(
                item.governed_statement_uuids
            )
        ],
    }


def source_pages_from_records(
    records: list[dict[str, object]],
) -> list[SourcePage]:
    """Hydrate ordered source pages and nested direct-image blocks."""
    return [
        SourcePage(
            index=page['index'],
            markdown=page['markdown'],
            blocks=[
                SourceBlock(
                    uuid=block['uuid'],
                    created_at=native_datetime(block['created_at']),
                    updated_at=native_datetime(block['updated_at']),
                    invalidated_at=native_datetime(block['invalidated_at'])
                    if block['invalidated_at'] is not None
                    else None,
                    block_type=block['block_type'],
                    content=block['content'],
                    origin_uuid=block.get('origin_uuid'),
                    embedding=block.get('embedding'),
                    crop_path=block.get('crop_path'),
                    crop_sha256=block.get('crop_sha256'),
                    crop_bbox=block.get('crop_bbox'),
                    image_path=block.get('image_path'),
                    image_sha256=block.get('image_sha256'),
                    image_page_indexes=block.get('image_page_indexes', []),
                )
                for block in page['blocks']
            ],
        )
        for page in records
    ]
