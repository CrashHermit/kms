"""Cypher statements for source pedagogical persistence."""

LOAD_SOURCE_PAGES = """
MATCH (source:Source {uuid: $source_uuid})
CALL (source) {
    OPTIONAL MATCH (source)-[:FIRST_PAGE]->(first_page:SourcePage)
    OPTIONAL MATCH page_path = (first_page)-[:NEXT_PAGE*0..]->(page:SourcePage)
    WITH page, page_path
    ORDER BY length(page_path)
    CALL (page) {
        OPTIONAL MATCH (page)-[:FIRST_BLOCK]->(first_block:SourceBlock)
        OPTIONAL MATCH block_path =
            (first_block)-[:NEXT_BLOCK*0..]->(block:SourceBlock)
        WHERE (page)-[:CONTAINS_BLOCK]->(block)
          AND all(path_block IN nodes(block_path)
                  WHERE (page)-[:CONTAINS_BLOCK]->(path_block))
        WITH block, block_path
        ORDER BY length(block_path)
        CALL (block) {
            OPTIONAL MATCH (block)-[:FIRST_VISUAL_ASSET]->(
                first_asset:VisualAsset
            )
            OPTIONAL MATCH asset_path =
                (first_asset)-[:NEXT_VISUAL_ASSET*0..]->(asset:VisualAsset)
            WHERE (block)-[:CONTAINS_VISUAL_ASSET]->(asset)
            WITH asset, asset_path
            ORDER BY length(asset_path)
            RETURN collect(
                CASE WHEN asset IS NULL THEN NULL
                ELSE {uuid: asset.uuid, path: asset.path} END
            ) AS raw_assets
        }
        RETURN collect(
            CASE WHEN block IS NULL THEN NULL
            ELSE {
                uuid: block.uuid,
                block_type: block.block_type,
                content: block.content,
                embedding: block.embedding,
                crop_path: block.crop_path,
                crop_bbox: block.crop_bbox,
                assets: [asset IN raw_assets WHERE asset IS NOT NULL]
            } END
        ) AS raw_blocks
    }
    RETURN collect(
        CASE WHEN page IS NULL THEN NULL
        ELSE {
            index: page.index,
            markdown: page.markdown,
            blocks: [block IN raw_blocks WHERE block IS NOT NULL]
        } END
    ) AS raw_pages
}
RETURN [page IN raw_pages WHERE page IS NOT NULL] AS pages
"""

REPLACE_SOURCE_PEDAGOGICAL_MATERIAL = """
MATCH (source:Source {uuid: $source_uuid})
CALL (source) {
    OPTIONAL MATCH (source)-[:HAS_PAGE]->(:SourcePage)
        -[:CONTAINS_BLOCK]->(:SourceBlock)-[:MEMBER_OF]->(old:Instruction)
    WITH collect(DISTINCT old) AS old_instructions
    FOREACH (item IN old_instructions | DETACH DELETE item)
    RETURN count(*) AS instructions_removed
}
CALL (source) {
    OPTIONAL MATCH (source)-[:HAS_PAGE]->(:SourcePage)
        -[:CONTAINS_BLOCK]->(:SourceBlock)-[:MEMBER_OF]->(
            :SourceStatement
        )-[:IN_SOURCE_HUB]->(statement_hub:SourceStatementHub)
    OPTIONAL MATCH (source)-[:HAS_PAGE]->(:SourcePage)
        -[:CONTAINS_BLOCK]->(:SourceBlock)-[:MEMBER_OF]->(
            :SourceProcedure
        )-[:IN_SOURCE_HUB]->(procedure_hub:SourceProcedureHub)
    WITH collect(DISTINCT statement_hub)
         + collect(DISTINCT procedure_hub) AS old_hubs
    FOREACH (hub IN old_hubs | DETACH DELETE hub)
    RETURN count(*) AS hubs_removed
}
CALL (source) {
    OPTIONAL MATCH (source)-[:HAS_PAGE]->(:SourcePage)
        -[:CONTAINS_BLOCK]->(:SourceBlock)-[:MEMBER_OF]->(old:SourceStatement)
    WITH collect(DISTINCT old) AS old_statements
    FOREACH (item IN old_statements | DETACH DELETE item)
    RETURN count(*) AS statements_removed
}
CALL (source) {
    OPTIONAL MATCH (source)-[:HAS_PAGE]->(:SourcePage)
        -[:CONTAINS_BLOCK]->(:SourceBlock)-[:MEMBER_OF]->(old:SourceProcedure)
    WITH collect(DISTINCT old) AS old_procedures
    FOREACH (item IN old_procedures | DETACH DELETE item)
    RETURN count(*) AS procedures_removed
}
CALL (source) {
    OPTIONAL MATCH (source)-[:HAS_PAGE]->(:SourcePage)
        -[:CONTAINS_BLOCK]->(parent:SourceBlock)
    WHERE NOT parent.uuid IN [row IN $blocks | row.uuid]
    OPTIONAL MATCH (parent)-[:CONTAINS_VISUAL_ASSET]->(asset:VisualAsset)
    OPTIONAL MATCH (retained_block:SourceBlock)
        -[:CONTAINS_VISUAL_ASSET]->(asset)
    WHERE retained_block.uuid IN [row IN $blocks | row.uuid]
    WITH collect(DISTINCT parent) AS parents,
         collect(DISTINCT CASE
             WHEN retained_block IS NULL THEN asset
         END) AS assets
    FOREACH (asset IN assets | DETACH DELETE asset)
    FOREACH (parent IN parents | DETACH DELETE parent)
    RETURN count(*) AS stale_blocks_removed
}
CALL (source) {
    OPTIONAL MATCH (source)-[first:FIRST_BLOCK]->(first_block)
    WHERE $first_block_uuid IS NULL
       OR first_block.uuid <> $first_block_uuid
    DELETE first
    WITH source
    OPTIONAL MATCH (source)-[last:LAST_BLOCK]->(last_block)
    WHERE $last_block_uuid IS NULL
       OR last_block.uuid <> $last_block_uuid
    DELETE last
    WITH source
    OPTIONAL MATCH (source)-[:HAS_PAGE]->(page:SourcePage)
    OPTIONAL MATCH (page)-[first:FIRST_BLOCK]->(first_block)
    WHERE NOT any(row IN $page_bounds
                  WHERE row.page_index = page.index
                    AND row.first_block_uuid = first_block.uuid)
    DELETE first
    WITH source
    OPTIONAL MATCH (source)-[:HAS_PAGE]->(page:SourcePage)
    OPTIONAL MATCH (page)-[last:LAST_BLOCK]->(last_block)
    WHERE NOT any(row IN $page_bounds
                  WHERE row.page_index = page.index
                    AND row.last_block_uuid = last_block.uuid)
    DELETE last
    WITH source
    OPTIONAL MATCH (source)-[:HAS_PAGE]->(page:SourcePage)
        -[membership:CONTAINS_BLOCK]->(block:SourceBlock)
    WHERE NOT any(row IN $pages
                  WHERE row.index = page.index
                    AND block.uuid IN row.block_uuids)
    DELETE membership
    WITH source
    OPTIONAL MATCH (source)-[:HAS_PAGE]->(:SourcePage)
        -[:CONTAINS_BLOCK]->(from_block:SourceBlock)
        -[next:NEXT_BLOCK]->(to_block:SourceBlock)
    WHERE NOT any(pair IN $block_pairs
                  WHERE pair.from_uuid = from_block.uuid
                    AND pair.to_uuid = to_block.uuid)
    DELETE next
    RETURN count(*) AS obsolete_links_removed
}
CALL (source) {
    UNWIND $blocks AS row
    OPTIONAL MATCH (source)-[:HAS_PAGE]->(:SourcePage)
        -[:CONTAINS_BLOCK]->(block:SourceBlock {uuid: row.uuid})
    WITH row, block
    WHERE block IS NULL
    CREATE (:SourceBlock {
        uuid: row.uuid,
        block_type: row.block_type,
        content: row.content,
        embedding: row.embedding,
        crop_path: row.crop_path,
        crop_bbox: row.crop_bbox
    })
    RETURN count(*) AS created_blocks
}
CALL (source) {
    UNWIND $pages AS row
    MATCH (source)-[:HAS_PAGE]->(page:SourcePage {index: row.index})
    UNWIND row.block_uuids AS block_uuid
    MATCH (block:SourceBlock {uuid: block_uuid})
    MERGE (page)-[:CONTAINS_BLOCK]->(block)
    RETURN count(*) AS block_memberships
}
CALL (source) {
    UNWIND $block_pairs AS pair
    MATCH (source)-[:HAS_PAGE]->(:SourcePage)
        -[:CONTAINS_BLOCK]->(from_block:SourceBlock {uuid: pair.from_uuid})
    MATCH (to_block:SourceBlock {uuid: pair.to_uuid})
    MERGE (from_block)-[:NEXT_BLOCK]->(to_block)
    RETURN count(*) AS block_links
}
CALL (source) {
    UNWIND $page_bounds AS row
    MATCH (source)-[:HAS_PAGE]->(page:SourcePage {index: row.page_index})
    MATCH (first_block:SourceBlock {uuid: row.first_block_uuid})
    MATCH (last_block:SourceBlock {uuid: row.last_block_uuid})
    MERGE (page)-[:FIRST_BLOCK]->(first_block)
    MERGE (page)-[:LAST_BLOCK]->(last_block)
    RETURN count(*) AS page_bounds
}
CALL (source) {
    WITH source WHERE $first_block_uuid IS NOT NULL
    MATCH (first_block:SourceBlock {uuid: $first_block_uuid})
    MERGE (source)-[:FIRST_BLOCK]->(first_block)
    RETURN count(*) AS source_first_bound
}
CALL (source) {
    WITH source WHERE $last_block_uuid IS NOT NULL
    MATCH (last_block:SourceBlock {uuid: $last_block_uuid})
    MERGE (source)-[:LAST_BLOCK]->(last_block)
    RETURN count(*) AS source_last_bound
}
CALL (source) {
    UNWIND $instructions AS row
    CREATE (instruction:Instruction {uuid: row.uuid})
    RETURN count(*) AS created_instructions
}
CALL (source) {
    UNWIND $instruction_member_pairs AS row
    MATCH (source)-[:HAS_PAGE]->(:SourcePage)
        -[:CONTAINS_BLOCK]->(block:SourceBlock {uuid: row.block_uuid})
    MATCH (instruction:Instruction {uuid: row.instruction_uuid})
    CREATE (block)-[:MEMBER_OF]->(instruction)
    RETURN count(*) AS instruction_memberships
}
CALL (source) {
    UNWIND $statements AS row
    CREATE (statement:SourceStatement {
        uuid: row.uuid,
        is_exercise: row.is_exercise,
        content: row.content,
        embedding: row.embedding
    })
    RETURN count(*) AS created_statements
}
CALL (source) {
    UNWIND $statement_member_pairs AS row
    MATCH (source)-[:HAS_PAGE]->(:SourcePage)
        -[:CONTAINS_BLOCK]->(block:SourceBlock {uuid: row.block_uuid})
    MATCH (statement:SourceStatement {uuid: row.statement_uuid})
    CREATE (block)-[:MEMBER_OF]->(statement)
    RETURN count(*) AS statement_memberships
}
CALL (source) {
    UNWIND $procedures AS row
    CREATE (procedure:SourceProcedure {
        uuid: row.uuid,
        content: row.content,
        embedding: row.embedding
    })
    RETURN count(*) AS created_procedures
}
CALL (source) {
    UNWIND $procedure_member_pairs AS row
    MATCH (source)-[:HAS_PAGE]->(:SourcePage)
        -[:CONTAINS_BLOCK]->(block:SourceBlock {uuid: row.block_uuid})
    MATCH (procedure:SourceProcedure {uuid: row.procedure_uuid})
    CREATE (block)-[:MEMBER_OF]->(procedure)
    RETURN count(*) AS procedure_memberships
}
CALL (source) {
    UNWIND $instruction_governance_pairs AS row
    MATCH (instruction:Instruction {uuid: row.instruction_uuid})
    MATCH (statement:SourceStatement {uuid: row.statement_uuid})
    CREATE (instruction)-[:GOVERNS]->(statement)
    RETURN count(*) AS governance
}
RETURN source.uuid AS uuid
"""
