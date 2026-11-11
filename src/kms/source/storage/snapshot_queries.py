"""Cypher statements for complete v3 owned source snapshots."""

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

LOAD_SOURCE_PAGES = """
MATCH (source:Source {uuid: $source_uuid, identity_schema: 'source-content-v3'})
CALL (source) {
    OPTIONAL MATCH (source)-[:FIRST_PAGE]->(first_page:SourcePage)
    OPTIONAL MATCH page_path = (first_page)-[:NEXT_PAGE*0..]->(page:SourcePage)
    WITH page, page_path ORDER BY length(page_path)
    CALL (page) {
        OPTIONAL MATCH (page)-[:FIRST_BLOCK]->(first_block:SourceBlock)
        OPTIONAL MATCH block_path = (first_block)-[:NEXT_BLOCK*0..]->(block:SourceBlock)
        WHERE (page)-[:CONTAINS_BLOCK]->(block)
          AND all(path_block IN nodes(block_path) WHERE (page)-[:CONTAINS_BLOCK]->(path_block))
        WITH block, block_path ORDER BY length(block_path)
        RETURN collect(CASE WHEN block IS NULL THEN NULL ELSE {
            uuid: block.uuid, created_at: block.created_at,
            updated_at: block.updated_at, invalidated_at: block.invalidated_at,
            block_type: block.block_type, content: block.content,
            origin_uuid: block.origin_uuid, embedding: block.embedding,
            crop_path: block.crop_path, crop_sha256: block.crop_sha256,
            crop_bbox: block.crop_bbox, image_path: block.image_path,
            image_sha256: block.image_sha256,
            image_page_indexes: coalesce(block.image_page_indexes, [])
        } END) AS raw_blocks
    }
    RETURN collect(CASE WHEN page IS NULL THEN NULL ELSE {
        index: page.index, markdown: page.markdown,
        created_at: page.created_at, updated_at: page.updated_at,
        blocks: [block IN raw_blocks WHERE block IS NOT NULL]
    } END) AS raw_pages
}
RETURN [page IN raw_pages WHERE page IS NOT NULL] AS pages
"""

READ_SOURCE_MATERIAL = """
MATCH (source:Source {uuid: $source_uuid, identity_schema: 'source-content-v3'})
CALL (source) {
    OPTIONAL MATCH (source)-[:HAS_PAGE]->(:SourcePage)-[:CONTAINS_BLOCK]->(:SourceBlock)
                  -[member:MEMBER_OF]->(instruction:Instruction)
    WITH DISTINCT instruction ORDER BY instruction.ordinal
    CALL (source, instruction) {
        OPTIONAL MATCH (source)-[:HAS_PAGE]->(:SourcePage)
                      -[:CONTAINS_BLOCK]->(block:SourceBlock)
                      -[membership:MEMBER_OF]->(instruction)
        WITH membership, block ORDER BY membership.position
        RETURN collect(block.uuid) AS member_uuids
    }
    CALL (instruction) {
        OPTIONAL MATCH (instruction)-[governance:GOVERNS]->(statement:SourceStatement)
        WITH governance, statement ORDER BY governance.position
        RETURN collect(statement.uuid) AS statement_uuids
    }
    RETURN collect(CASE WHEN instruction IS NULL THEN NULL ELSE {
        uuid: instruction.uuid, created_at: instruction.created_at,
        updated_at: instruction.updated_at,
        invalidated_at: instruction.invalidated_at,
        member_block_uuids: member_uuids,
        governed_statement_uuids: statement_uuids
    } END) AS rows
}
WITH source, [row IN rows WHERE row IS NOT NULL] AS instruction_rows
CALL (source) {
    OPTIONAL MATCH (source)-[:HAS_PAGE]->(:SourcePage)-[:CONTAINS_BLOCK]->(:SourceBlock)
                  -[:MEMBER_OF]->(statement:SourceStatement)
    WITH DISTINCT statement ORDER BY statement.ordinal
    CALL (source, statement) {
        OPTIONAL MATCH (source)-[:HAS_PAGE]->(:SourcePage)
                      -[:CONTAINS_BLOCK]->(block:SourceBlock)
                      -[membership:MEMBER_OF]->(statement)
        WITH membership, block ORDER BY membership.position
        RETURN collect(block.uuid) AS member_uuids
    }
    RETURN collect(CASE WHEN statement IS NULL THEN NULL ELSE {
        uuid: statement.uuid, content: statement.content,
        is_exercise: statement.is_exercise, embedding: statement.embedding,
        member_block_uuids: member_uuids
    } END) AS rows
}
WITH source, instruction_rows, [row IN rows WHERE row IS NOT NULL] AS statement_rows
CALL (source) {
    OPTIONAL MATCH (source)-[:HAS_PAGE]->(:SourcePage)-[:CONTAINS_BLOCK]->(:SourceBlock)
                  -[:MEMBER_OF]->(procedure:SourceProcedure)
    WITH DISTINCT procedure ORDER BY procedure.ordinal
    CALL (source, procedure) {
        OPTIONAL MATCH (source)-[:HAS_PAGE]->(:SourcePage)
                      -[:CONTAINS_BLOCK]->(block:SourceBlock)
                      -[membership:MEMBER_OF]->(procedure)
        WITH membership, block ORDER BY membership.position
        RETURN collect(block.uuid) AS member_uuids
    }
    RETURN collect(CASE WHEN procedure IS NULL THEN NULL ELSE {
        uuid: procedure.uuid, content: procedure.content,
        embedding: procedure.embedding, member_block_uuids: member_uuids
    } END) AS rows
}
RETURN instruction_rows AS instructions, statement_rows AS statements,
       [row IN rows WHERE row IS NOT NULL] AS procedures
"""
