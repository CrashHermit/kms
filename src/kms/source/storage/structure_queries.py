"""Non-destructive structural creation shared by owned source snapshots."""

CREATE_SOURCE_STRUCTURE = """
CALL (source) {
    UNWIND $units AS row
    CREATE (unit:SourceUnit {
        uuid: row.uuid, kind: row.kind, embedding: row.embedding,
        created_at: row.created_at, updated_at: row.updated_at,
        invalidated_at: row.invalidated_at
    })
    CREATE (source)-[:HAS_UNIT {
        created_at: $edge_created_at, updated_at: $edge_created_at
    }]->(unit)
    RETURN count(*) AS _
}
WITH source
CALL (source) {
    UNWIND $blocks AS row
    CREATE (block:SourceBlock {
        uuid: row.uuid, created_at: row.created_at,
        updated_at: row.updated_at, invalidated_at: row.invalidated_at,
        block_type: row.block_type, content: row.content,
        embedding: row.embedding, crop_path: row.crop_path,
        crop_sha256: row.crop_sha256, crop_bbox: row.crop_bbox,
        image_path: row.image_path, image_sha256: row.image_sha256
    })
    RETURN count(*) AS _
}
WITH source
CALL (source) {
    UNWIND $unit_block_pairs AS row
    MATCH (source)-[:HAS_UNIT]->(unit:SourceUnit {uuid: row.unit_uuid})
    MATCH (block:SourceBlock {uuid: row.block_uuid})
    CREATE (unit)-[:HAS_BLOCK {
        created_at: $edge_created_at, updated_at: $edge_created_at
    }]->(block)
    RETURN count(*) AS _
}
WITH source
CALL (source) {
    UNWIND $unit_pairs AS row
    MATCH (source)-[:HAS_UNIT]->(from_unit:SourceUnit {uuid: row.from_uuid})
    MATCH (source)-[:HAS_UNIT]->(to_unit:SourceUnit {uuid: row.to_uuid})
    CREATE (from_unit)-[:NEXT_UNIT {
        created_at: $edge_created_at, updated_at: $edge_created_at
    }]->(to_unit)
    RETURN count(*) AS _
}
WITH source
CALL (source) {
    UNWIND $block_pairs AS row
    MATCH (source)-[:HAS_UNIT]->(unit:SourceUnit {uuid: row.unit_uuid})
    MATCH (unit)-[:HAS_BLOCK]->(from_block:SourceBlock {uuid: row.from_uuid})
    MATCH (unit)-[:HAS_BLOCK]->(to_block:SourceBlock {uuid: row.to_uuid})
    CREATE (from_block)-[:NEXT_BLOCK {
        created_at: $edge_created_at, updated_at: $edge_created_at
    }]->(to_block)
    RETURN count(*) AS _
}
WITH source
CALL (source) {
    UNWIND $unit_bounds AS row
    MATCH (source)-[:HAS_UNIT]->(unit:SourceUnit {uuid: row.unit_uuid})
    MATCH (first_block:SourceBlock {uuid: row.first_block_uuid})
    MATCH (last_block:SourceBlock {uuid: row.last_block_uuid})
    CREATE (unit)-[:FIRST_BLOCK {
        created_at: $edge_created_at, updated_at: $edge_created_at
    }]->(first_block)
    CREATE (unit)-[:LAST_BLOCK {
        created_at: $edge_created_at, updated_at: $edge_created_at
    }]->(last_block)
    RETURN count(*) AS _
}
WITH source
CALL (source) {
    WITH source WHERE $first_unit_uuid IS NOT NULL
    MATCH (first_unit:SourceUnit {uuid: $first_unit_uuid})
    CREATE (source)-[:FIRST_UNIT {
        created_at: $edge_created_at, updated_at: $edge_created_at
    }]->(first_unit)
    RETURN count(*) AS _
}
WITH source
CALL (source) {
    WITH source WHERE $last_unit_uuid IS NOT NULL
    MATCH (last_unit:SourceUnit {uuid: $last_unit_uuid})
    CREATE (source)-[:LAST_UNIT {
        created_at: $edge_created_at, updated_at: $edge_created_at
    }]->(last_unit)
    RETURN count(*) AS _
}
"""
