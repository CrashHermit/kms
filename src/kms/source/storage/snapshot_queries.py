"""Cypher statements for complete v5 owned source snapshots."""

from kms.source.storage.structure_queries import CREATE_SOURCE_STRUCTURE

LOCK_SNAPSHOT_OWNER = """
MATCH (user:User {uuid: $user_uuid})
SET user._source_snapshot_lock = randomUUID()
REMOVE user._source_snapshot_lock
RETURN user.uuid AS uuid
"""

READ_OWNED_SNAPSHOT = """
MATCH (:User {uuid: $user_uuid})-[:OWNS_SOURCE]->(source:Source {uuid: $source_uuid})
RETURN source.uuid AS uuid, source.key AS key, source.metadata AS metadata,
       source.identity_schema AS identity_schema,
       source.created_at AS created_at, source.updated_at AS updated_at,
       source.invalidated_at AS invalidated_at,
       coalesce(source.revision, 0) AS revision
"""

CREATE_OWNED_SNAPSHOT = """
MATCH (user:User {uuid: $user_uuid})
CREATE (source:Source {
    uuid: $source.uuid, key: $source.key, metadata: $source.metadata,
    identity_schema: $source.identity_schema, revision: 0,
    created_at: $source.created_at, updated_at: $source.updated_at,
    invalidated_at: $source.invalidated_at
})
CREATE (user)-[:OWNS_SOURCE {
    created_at: $edge_created_at, updated_at: $edge_created_at
}]->(source)
WITH source
__CREATE_SOURCE_STRUCTURE__
RETURN source.uuid AS uuid
""".replace('__CREATE_SOURCE_STRUCTURE__', CREATE_SOURCE_STRUCTURE)

READ_SOURCE_UNITS = """
MATCH (source:Source {uuid: $source_uuid, identity_schema: $identity_schema})
CALL (source) {
    OPTIONAL MATCH (source)-[:FIRST_UNIT]->(first_unit:SourceUnit)
    OPTIONAL MATCH unit_path = (first_unit)-[:NEXT_UNIT*0..]->(unit:SourceUnit)
    WHERE (source)-[:HAS_UNIT]->(unit)
      AND all(path_unit IN nodes(unit_path) WHERE (source)-[:HAS_UNIT]->(path_unit))
    WITH unit, unit_path ORDER BY length(unit_path)
    CALL (unit) {
        OPTIONAL MATCH (unit)-[:FIRST_BLOCK]->(first_block:SourceBlock)
        OPTIONAL MATCH block_path = (first_block)-[:NEXT_BLOCK*0..]->(block:SourceBlock)
        WHERE (unit)-[:HAS_BLOCK]->(block)
          AND all(path_block IN nodes(block_path) WHERE (unit)-[:HAS_BLOCK]->(path_block))
        WITH block, block_path ORDER BY length(block_path)
        RETURN collect(CASE WHEN block IS NULL THEN NULL ELSE {
            uuid: block.uuid, created_at: block.created_at,
            updated_at: block.updated_at, invalidated_at: block.invalidated_at,
            block_type: block.block_type, content: block.content,
            embedding: block.embedding,
            crop_path: block.crop_path, crop_sha256: block.crop_sha256,
            crop_bbox: block.crop_bbox, image_path: block.image_path,
            image_sha256: block.image_sha256
        } END) AS raw_blocks
    }
    RETURN collect(CASE WHEN unit IS NULL THEN NULL ELSE {
        uuid: unit.uuid, kind: unit.kind, embedding: unit.embedding,
        created_at: unit.created_at, updated_at: unit.updated_at,
        invalidated_at: unit.invalidated_at,
        blocks: [block IN raw_blocks WHERE block IS NOT NULL]
    } END) AS raw_units
}
RETURN [unit IN raw_units WHERE unit IS NOT NULL] AS units
"""
