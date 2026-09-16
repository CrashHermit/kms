"""Cypher statements for loading and replacing source graphs."""

READ_SOURCES = """
MATCH (source:Source)
RETURN source.uuid AS uuid, source.key AS key
ORDER BY source.key, source.uuid
"""

REPLACE_SOURCE = """
MERGE (source:Source {uuid: $source.uuid})
SET source.key = $source.key,
    source.metadata = $source.metadata
WITH source
OPTIONAL MATCH (source)-[:HAS_PAGE]->(old_page:SourcePage)
OPTIONAL MATCH (old_page)-[:CONTAINS_BLOCK]->(old_block:SourceBlock)
OPTIONAL MATCH (old_block)-[:MEMBER_OF]->(old_statement:Statement)
OPTIONAL MATCH (old_block)-[:MEMBER_OF]->(old_procedure:Procedure)
OPTIONAL MATCH (old_block)-[:CONTAINS_VISUAL_ASSET]->(old_asset:VisualAsset)
OPTIONAL MATCH (source)-[:HAS_INSTRUCTION]->(old_instruction:Instruction)
WITH source,
     collect(DISTINCT old_page) AS old_pages,
     collect(DISTINCT old_block) AS old_blocks,
     collect(DISTINCT old_statement) AS old_statements,
     collect(DISTINCT old_procedure) AS old_procedures,
     collect(DISTINCT old_asset) AS old_assets,
     collect(DISTINCT old_instruction) AS old_instructions
FOREACH (page IN old_pages | DETACH DELETE page)
FOREACH (block IN old_blocks | DETACH DELETE block)
FOREACH (statement IN old_statements | DETACH DELETE statement)
FOREACH (procedure IN old_procedures | DETACH DELETE procedure)
FOREACH (asset IN old_assets | DETACH DELETE asset)
FOREACH (instruction IN old_instructions | DETACH DELETE instruction)
WITH source
CALL (source) {
    UNWIND $pages AS row
    CREATE (page:SourcePage {index: row.index, markdown: row.markdown})
    CREATE (source)-[:HAS_PAGE]->(page)
    RETURN count(*) AS _
}
WITH source
CALL (source) {
    UNWIND $blocks AS row
    CREATE (block:SourceBlock {
        uuid: row.uuid,
        block_type: row.block_type,
        content: row.content,
        embedding: row.embedding,
        crop_path: row.crop_path,
        crop_bbox: row.crop_bbox
    })
    FOREACH (_ IN CASE WHEN row.block_type = 'equation' THEN [1] ELSE [] END | SET block:Equation)
    FOREACH (_ IN CASE WHEN row.block_type = 'paragraph' THEN [1] ELSE [] END | SET block:Paragraph)
    FOREACH (_ IN CASE WHEN row.block_type = 'math' THEN [1] ELSE [] END | SET block:Math)
    FOREACH (_ IN CASE WHEN row.block_type = 'code' THEN [1] ELSE [] END | SET block:Code)
    FOREACH (_ IN CASE WHEN row.block_type = 'list' THEN [1] ELSE [] END | SET block:List)
    FOREACH (_ IN CASE WHEN row.block_type = 'table' THEN [1] ELSE [] END | SET block:Table)
    FOREACH (_ IN CASE WHEN row.block_type = 'image' THEN [1] ELSE [] END | SET block:Image)
    FOREACH (_ IN CASE WHEN row.block_type = 'caption' THEN [1] ELSE [] END | SET block:Caption)
    FOREACH (_ IN CASE WHEN row.block_type = 'header' THEN [1] ELSE [] END | SET block:Header)
    FOREACH (_ IN CASE WHEN row.block_type = 'bibliographic' THEN [1] ELSE [] END | SET block:Bibliographic)
    FOREACH (_ IN CASE WHEN row.block_type = 'note' THEN [1] ELSE [] END | SET block:Note)
    FOREACH (_ IN CASE WHEN row.block_type = 'footer' THEN [1] ELSE [] END | SET block:Footer)
    FOREACH (_ IN CASE WHEN row.block_type = 'markdown' THEN [1] ELSE [] END | SET block:Markdown)
    FOREACH (_ IN CASE WHEN row.block_type = 'aside_text' THEN [1] ELSE [] END | SET block:AsideText)
    FOREACH (_ IN CASE WHEN row.block_type = 'instruction' THEN [1] ELSE [] END | SET block:Instruction)
    RETURN count(*) AS _
}
WITH source
CALL (source) {
    UNWIND $assets AS row
    CREATE (asset:VisualAsset {uuid: row.uuid, path: row.path})
    RETURN count(*) AS _
}
WITH source
CALL (source) {
    UNWIND $block_asset_pairs AS row
    MATCH (block:SourceBlock {uuid: row.block_uuid})
    MATCH (asset:VisualAsset {uuid: row.asset_uuid})
    CREATE (block)-[:CONTAINS_VISUAL_ASSET]->(asset)
    RETURN count(*) AS _
}
WITH source
CALL (source) {
    UNWIND $asset_pairs AS row
    MATCH (from_asset:VisualAsset {uuid: row.from_uuid})
    MATCH (to_asset:VisualAsset {uuid: row.to_uuid})
    CREATE (from_asset)-[:NEXT_VISUAL_ASSET]->(to_asset)
    RETURN count(*) AS _
}
WITH source
CALL (source) {
    UNWIND $asset_bounds AS row
    MATCH (block:SourceBlock {uuid: row.block_uuid})
    MATCH (first_asset:VisualAsset {uuid: row.first_asset_uuid})
    MATCH (last_asset:VisualAsset {uuid: row.last_asset_uuid})
    CREATE (block)-[:FIRST_VISUAL_ASSET]->(first_asset)
    CREATE (block)-[:LAST_VISUAL_ASSET]->(last_asset)
    RETURN count(*) AS _
}
WITH source
CALL (source) {
    UNWIND $page_block_pairs AS row
    MATCH (source)-[:HAS_PAGE]->(page:SourcePage {index: row.page_index})
    MATCH (block:SourceBlock {uuid: row.block_uuid})
    CREATE (page)-[:CONTAINS_BLOCK]->(block)
    RETURN count(*) AS _
}
WITH source
CALL (source) {
    UNWIND $page_pairs AS row
    MATCH (source)-[:HAS_PAGE]->(from_page:SourcePage {index: row.from_index})
    MATCH (source)-[:HAS_PAGE]->(to_page:SourcePage {index: row.to_index})
    CREATE (from_page)-[:NEXT_PAGE]->(to_page)
    RETURN count(*) AS _
}
WITH source
CALL (source) {
    UNWIND $block_pairs AS row
    MATCH (from_block:SourceBlock {uuid: row.from_uuid})
    MATCH (to_block:SourceBlock {uuid: row.to_uuid})
    CREATE (from_block)-[:NEXT_BLOCK]->(to_block)
    RETURN count(*) AS _
}
WITH source
CALL (source) {
    UNWIND $instructions AS row
    CREATE (instruction:Instruction {uuid: row.uuid})
    CREATE (source)-[:HAS_INSTRUCTION]->(instruction)
    RETURN count(*) AS _
}
WITH source
CALL (source) {
    UNWIND $instruction_member_pairs AS row
    MATCH (block:SourceBlock {uuid: row.block_uuid})
    MATCH (instruction:Instruction {uuid: row.instruction_uuid})
    CREATE (block)-[:MEMBER_OF]->(instruction)
    RETURN count(*) AS _
}
WITH source
CALL (source) {
    UNWIND $statements AS row
    CREATE (statement:Statement {
        uuid: row.uuid,
        source_uuid: row.source_uuid,
        is_exercise: row.is_exercise
    })
    RETURN count(*) AS _
}
WITH source
CALL (source) {
    UNWIND $procedures AS row
    CREATE (procedure:Procedure {
        uuid: row.uuid,
        source_uuid: row.source_uuid
    })
    RETURN count(*) AS _
}
WITH source
CALL (source) {
    UNWIND $statement_member_pairs AS row
    MATCH (block:SourceBlock {uuid: row.block_uuid})
    MATCH (statement:Statement {uuid: row.statement_uuid})
    CREATE (block)-[:MEMBER_OF]->(statement)
    RETURN count(*) AS _
}
WITH source
CALL (source) {
    UNWIND $procedure_member_pairs AS row
    MATCH (block:SourceBlock {uuid: row.block_uuid})
    MATCH (procedure:Procedure {uuid: row.procedure_uuid})
    CREATE (block)-[:MEMBER_OF]->(procedure)
    RETURN count(*) AS _
}
WITH source
CALL (source) {
    UNWIND $instruction_governance_pairs AS row
    MATCH (instruction:Instruction {uuid: row.instruction_uuid})
    MATCH (statement:Statement {uuid: row.statement_uuid})
    CREATE (instruction)-[:GOVERNS]->(statement)
    RETURN count(*) AS _
}
WITH source
CALL (source) {
    UNWIND $page_bounds AS row
    MATCH (source)-[:HAS_PAGE]->(page:SourcePage {index: row.page_index})
    MATCH (first_block:SourceBlock {uuid: row.first_block_uuid})
    MATCH (last_block:SourceBlock {uuid: row.last_block_uuid})
    CREATE (page)-[:FIRST_BLOCK]->(first_block)
    CREATE (page)-[:LAST_BLOCK]->(last_block)
    RETURN count(*) AS _
}
WITH source
CALL (source) {
    WITH source WHERE $first_page_index IS NOT NULL
    MATCH (source)-[:HAS_PAGE]->(page:SourcePage {index: $first_page_index})
    CREATE (source)-[:FIRST_PAGE]->(page)
    RETURN count(*) AS _
}
WITH source
CALL (source) {
    WITH source WHERE $first_block_uuid IS NOT NULL
    MATCH (first_block:SourceBlock {uuid: $first_block_uuid})
    CREATE (source)-[:FIRST_BLOCK]->(first_block)
    RETURN count(*) AS _
}
WITH source
CALL (source) {
    WITH source WHERE $last_block_uuid IS NOT NULL
    MATCH (last_block:SourceBlock {uuid: $last_block_uuid})
    CREATE (source)-[:LAST_BLOCK]->(last_block)
    RETURN count(*) AS _
}
RETURN source.uuid AS uuid
"""

READ_SOURCE_BLOCKS = """
MATCH (source:Source {uuid: $source_uuid})-[:FIRST_BLOCK]->(first:SourceBlock)
MATCH path = (first)-[:NEXT_BLOCK*0..]->(block:SourceBlock)
RETURN block.uuid AS uuid, block.block_type AS block_type, block.content AS content
ORDER BY length(path)
"""

FIND_SIMILAR_SOURCE_BLOCKS = """
MATCH (query:SourceBlock {uuid: $query_uuid})
CALL db.index.vector.queryNodes(
    'source_block_embedding',
    $candidate_limit,
    query.embedding
)
YIELD node, score
WHERE node.uuid <> query.uuid
RETURN node.uuid AS uuid,
       node.block_type AS block_type,
       node.content AS content,
       score
ORDER BY score DESC, uuid ASC
LIMIT $top_k
"""
