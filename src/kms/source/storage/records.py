"""Shared serialization and hydration for structural source graphs."""

import json
from datetime import datetime
from typing import Any

from kms.core.models import Edge
from kms.source.models.snapshot import SourceSnapshot
from kms.source.models.source import SourceBlock
from kms.source.models.unit import SourceUnit


def native_datetime(value: Any) -> datetime:
    """Convert a Neo4j temporal value at the database boundary."""
    return value if isinstance(value, datetime) else value.to_native()


def source_graph_parameters(snapshot: SourceSnapshot) -> dict[str, object]:
    """Serialize a complete source snapshot for one structural transaction."""
    source, units = snapshot.source, snapshot.units
    blocks = [block for unit in units for block in unit.blocks]
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
        'units': [
            {
                'uuid': unit.uuid,
                'kind': unit.kind.value,
                'embedding': unit.embedding,
                'created_at': unit.created_at,
                'updated_at': unit.updated_at,
                'invalidated_at': unit.invalidated_at,
            }
            for unit in units
        ],
        'blocks': [
            {
                'uuid': block.uuid,
                'created_at': block.created_at,
                'updated_at': block.updated_at,
                'invalidated_at': block.invalidated_at,
                'block_type': block.block_type.value,
                'content': block.content,
                'embedding': block.embedding,
                'crop_path': block.crop_path,
                'crop_sha256': block.crop_sha256,
                'crop_bbox': list(block.crop_bbox)
                if block.crop_bbox is not None
                else None,
                'image_path': block.image_path,
                'image_sha256': block.image_sha256,
            }
            for block in blocks
        ],
        'unit_block_pairs': [
            {'unit_uuid': unit.uuid, 'block_uuid': block.uuid}
            for unit in units
            for block in unit.blocks
        ],
        'unit_pairs': [
            {'from_uuid': unit.uuid, 'to_uuid': nxt.uuid}
            for unit, nxt in zip(units, units[1:], strict=False)
        ],
        'block_pairs': [
            {
                'unit_uuid': unit.uuid,
                'from_uuid': block.uuid,
                'to_uuid': nxt.uuid,
            }
            for unit in units
            for block, nxt in zip(unit.blocks, unit.blocks[1:], strict=False)
        ],
        'unit_bounds': [
            {
                'unit_uuid': unit.uuid,
                'first_block_uuid': unit.blocks[0].uuid,
                'last_block_uuid': unit.blocks[-1].uuid,
            }
            for unit in units
        ],
        'first_unit_uuid': units[0].uuid if units else None,
        'last_unit_uuid': units[-1].uuid if units else None,
    }


def source_units_from_records(
    records: list[dict[str, object]],
) -> list[SourceUnit]:
    """Hydrate ordered units and their ordered direct-containment blocks."""
    units: list[SourceUnit] = []
    for unit in records:
        units.append(
            SourceUnit(
                uuid=unit['uuid'],
                created_at=native_datetime(unit['created_at']),
                updated_at=native_datetime(unit['updated_at']),
                invalidated_at=native_datetime(unit['invalidated_at'])
                if unit['invalidated_at'] is not None
                else None,
                kind=unit['kind'],
                embedding=unit.get('embedding'),
                blocks=[
                    SourceBlock(
                        uuid=block['uuid'],
                        created_at=native_datetime(block['created_at']),
                        updated_at=native_datetime(block['updated_at']),
                        invalidated_at=native_datetime(block['invalidated_at'])
                        if block['invalidated_at'] is not None
                        else None,
                        block_type=block['block_type'],
                        content=block.get('content'),
                        embedding=block.get('embedding'),
                        crop_path=block.get('crop_path'),
                        crop_sha256=block.get('crop_sha256'),
                        crop_bbox=block.get('crop_bbox'),
                        image_path=block.get('image_path'),
                        image_sha256=block.get('image_sha256'),
                    )
                    for block in unit['blocks']
                ],
            )
        )
    return units
