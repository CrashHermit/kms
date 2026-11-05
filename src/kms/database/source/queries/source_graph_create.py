"""Non-destructive structural creation shared by imports and snapshots."""

CREATE_SOURCE_STRUCTURE = """
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
        origin_uuid: row.origin_uuid,
        embedding: row.embedding,
        crop_path: row.crop_path,
        crop_sha256: row.crop_sha256,
        crop_bbox: row.crop_bbox
    })
    RETURN count(*) AS _
}
WITH source
CALL (source) {
    UNWIND $assets AS row
    CREATE (asset:VisualAsset {
        uuid: row.uuid,
        path: row.path,
        content_sha256: row.content_sha256
    })
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
    MATCH (source)-[:HAS_PAGE]->(:SourcePage)-[:CONTAINS_BLOCK]->(last_block)
    CREATE (source)-[:LAST_BLOCK]->(last_block)
    RETURN count(*) AS _
}
"""
