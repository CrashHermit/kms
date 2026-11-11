"""Non-destructive structural creation shared by owned source snapshots."""

CREATE_SOURCE_STRUCTURE = """
CALL (source) {
    UNWIND $pages AS row
    CREATE (page:SourcePage {
        index: row.index, markdown: row.markdown,
        created_at: datetime.transaction(),
        updated_at: datetime.transaction()
    })
    CREATE (source)-[:HAS_PAGE {
        created_at: $edge_created_at, updated_at: $edge_created_at
    }]->(page)
    RETURN count(*) AS _
}
WITH source
CALL (source) {
    UNWIND $blocks AS row
    CREATE (block:SourceBlock {
        uuid: row.uuid, created_at: row.created_at,
        updated_at: row.updated_at, invalidated_at: row.invalidated_at,
        block_type: row.block_type, content: row.content,
        origin_uuid: row.origin_uuid, embedding: row.embedding,
        crop_path: row.crop_path, crop_sha256: row.crop_sha256,
        crop_bbox: row.crop_bbox, image_path: row.image_path,
        image_sha256: row.image_sha256, image_page_indexes: row.image_page_indexes
    })
    RETURN count(*) AS _
}
WITH source
CALL (source) {
    UNWIND $page_block_pairs AS row
    MATCH (source)-[:HAS_PAGE]->(page:SourcePage {index: row.page_index})
    MATCH (block:SourceBlock {uuid: row.block_uuid})
    CREATE (page)-[:CONTAINS_BLOCK {
        created_at: $edge_created_at, updated_at: $edge_created_at
    }]->(block)
    RETURN count(*) AS _
}
WITH source
CALL (source) {
    UNWIND $instructions AS row
    CREATE (:Instruction {
        uuid: row.uuid, ordinal: row.ordinal, created_at: row.created_at,
        updated_at: row.updated_at, invalidated_at: row.invalidated_at
    })
    RETURN count(*) AS _
}
WITH source
CALL (source) {
    UNWIND $statements AS row
    CREATE (:SourceStatement {
        uuid: row.uuid, content: row.content, is_exercise: row.is_exercise,
        embedding: row.embedding, ordinal: row.ordinal,
        created_at: datetime.transaction(), updated_at: datetime.transaction()
    })
    RETURN count(*) AS _
}
WITH source
CALL (source) {
    UNWIND $procedures AS row
    CREATE (:SourceProcedure {
        uuid: row.uuid, content: row.content, embedding: row.embedding,
        ordinal: row.ordinal, created_at: datetime.transaction(),
        updated_at: datetime.transaction()
    })
    RETURN count(*) AS _
}
WITH source
CALL (source) {
    UNWIND $instruction_member_pairs AS row
    MATCH (source)-[:HAS_PAGE]->(:SourcePage)-[:CONTAINS_BLOCK]->(block:SourceBlock {uuid: row.block_uuid})
    MATCH (instruction:Instruction {uuid: row.instruction_uuid})
    CREATE (block)-[:MEMBER_OF {
        position: row.position, created_at: $edge_created_at,
        updated_at: $edge_created_at
    }]->(instruction)
    RETURN count(*) AS _
}
WITH source
CALL (source) {
    UNWIND $statement_member_pairs AS row
    MATCH (source)-[:HAS_PAGE]->(:SourcePage)-[:CONTAINS_BLOCK]->(block:SourceBlock {uuid: row.block_uuid})
    MATCH (statement:SourceStatement {uuid: row.statement_uuid})
    CREATE (block)-[:MEMBER_OF {
        position: row.position, created_at: $edge_created_at,
        updated_at: $edge_created_at
    }]->(statement)
    RETURN count(*) AS _
}
WITH source
CALL (source) {
    UNWIND $procedure_member_pairs AS row
    MATCH (source)-[:HAS_PAGE]->(:SourcePage)-[:CONTAINS_BLOCK]->(block:SourceBlock {uuid: row.block_uuid})
    MATCH (procedure:SourceProcedure {uuid: row.procedure_uuid})
    CREATE (block)-[:MEMBER_OF {
        position: row.position, created_at: $edge_created_at,
        updated_at: $edge_created_at
    }]->(procedure)
    RETURN count(*) AS _
}
WITH source
CALL (source) {
    UNWIND $instruction_governance_pairs AS row
    MATCH (instruction:Instruction {uuid: row.instruction_uuid})
    MATCH (statement:SourceStatement {uuid: row.statement_uuid})
    CREATE (instruction)-[:GOVERNS {
        position: row.position, created_at: $edge_created_at,
        updated_at: $edge_created_at
    }]->(statement)
    RETURN count(*) AS _
}
WITH source
CALL (source) {
    UNWIND $page_pairs AS row
    MATCH (source)-[:HAS_PAGE]->(from_page:SourcePage {index: row.from_index})
    MATCH (source)-[:HAS_PAGE]->(to_page:SourcePage {index: row.to_index})
    CREATE (from_page)-[:NEXT_PAGE {
        created_at: $edge_created_at, updated_at: $edge_created_at
    }]->(to_page)
    RETURN count(*) AS _
}
WITH source
CALL (source) {
    UNWIND $block_pairs AS row
    MATCH (from_block:SourceBlock {uuid: row.from_uuid})
    MATCH (to_block:SourceBlock {uuid: row.to_uuid})
    CREATE (from_block)-[:NEXT_BLOCK {
        created_at: $edge_created_at, updated_at: $edge_created_at
    }]->(to_block)
    RETURN count(*) AS _
}
WITH source
CALL (source) {
    UNWIND $page_bounds AS row
    MATCH (source)-[:HAS_PAGE]->(page:SourcePage {index: row.page_index})
    MATCH (first_block:SourceBlock {uuid: row.first_block_uuid})
    MATCH (last_block:SourceBlock {uuid: row.last_block_uuid})
    CREATE (page)-[:FIRST_BLOCK {
        created_at: $edge_created_at, updated_at: $edge_created_at
    }]->(first_block)
    CREATE (page)-[:LAST_BLOCK {
        created_at: $edge_created_at, updated_at: $edge_created_at
    }]->(last_block)
    RETURN count(*) AS _
}
WITH source
CALL (source) {
    WITH source WHERE $first_page_index IS NOT NULL
    MATCH (source)-[:HAS_PAGE]->(page:SourcePage {index: $first_page_index})
    CREATE (source)-[:FIRST_PAGE {
        created_at: $edge_created_at, updated_at: $edge_created_at
    }]->(page)
    RETURN count(*) AS _
}
WITH source
CALL (source) {
    WITH source WHERE $first_block_uuid IS NOT NULL
    MATCH (first_block:SourceBlock {uuid: $first_block_uuid})
    CREATE (source)-[:FIRST_BLOCK {
        created_at: $edge_created_at, updated_at: $edge_created_at
    }]->(first_block)
    RETURN count(*) AS _
}
WITH source
CALL (source) {
    WITH source WHERE $last_block_uuid IS NOT NULL
    MATCH (last_block:SourceBlock {uuid: $last_block_uuid})
    MATCH (source)-[:HAS_PAGE]->(:SourcePage)-[:CONTAINS_BLOCK]->(last_block)
    CREATE (source)-[:LAST_BLOCK {
        created_at: $edge_created_at, updated_at: $edge_created_at
    }]->(last_block)
    RETURN count(*) AS _
}
"""
