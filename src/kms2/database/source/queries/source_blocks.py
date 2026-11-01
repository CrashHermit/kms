"""Cypher statements for ordered source-block access and search."""

READ_SOURCE_BLOCKS = """
MATCH (source:Source {uuid: $source_uuid})-[:FIRST_BLOCK]->(first:SourceBlock)
MATCH path = (first)-[:NEXT_BLOCK*0..]->(block:SourceBlock)
RETURN block.uuid AS uuid,
       block.block_type AS block_type,
       block.content AS content
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

SEARCH_OWNED_SOURCE_BLOCKS = """
MATCH (:User {uuid: $user_uuid})-[:OWNS_SOURCE]->(source:Source)
      -[:HAS_PAGE]->(:SourcePage)-[:CONTAINS_BLOCK]->(block:SourceBlock)
WHERE ($source_uuids IS NULL OR source.uuid IN $source_uuids)
  AND block.content IS NOT NULL
  AND block.embedding IS NOT NULL
WITH DISTINCT source, block
WITH source, block,
     vector.similarity.cosine(block.embedding, $query_embedding) AS similarity
ORDER BY similarity DESC, block.uuid ASC
LIMIT $candidate_limit
CALL (source, block) {
    OPTIONAL MATCH (source)-[:HAS_PAGE]->(page:SourcePage)
          -[:CONTAINS_BLOCK]->(block)
    WITH DISTINCT page.index AS page_index
    ORDER BY page_index
    RETURN collect(page_index) AS page_indexes
}
CALL (block) {
    OPTIONAL MATCH (block)-[:FIRST_VISUAL_ASSET]->(first_asset:VisualAsset)
    OPTIONAL MATCH asset_path = (first_asset)
        -[:NEXT_VISUAL_ASSET*0..]->(asset:VisualAsset)
    WHERE (block)-[:CONTAINS_VISUAL_ASSET]->(asset)
    WITH asset, asset_path
    ORDER BY length(asset_path)
    RETURN [
        item IN collect(
            CASE WHEN asset IS NULL THEN NULL
            ELSE {uuid: asset.uuid, path: asset.path} END
        )
        WHERE item IS NOT NULL
    ] AS assets
}
RETURN {
    uuid: source.uuid,
    key: source.key,
    metadata: source.metadata,
    revision: coalesce(source.revision, 0)
} AS source,
{
    uuid: block.uuid,
    block_type: block.block_type,
    content: block.content,
    crop_path: block.crop_path,
    crop_bbox: block.crop_bbox,
    assets: assets
} AS block,
page_indexes,
similarity
"""
SEARCH_OWNED_SOURCE_BLOCKS_VECTOR = """
CALL db.index.vector.queryNodes(
    'source_block_embedding',
    $candidate_limit,
    $query_embedding
)
YIELD node AS block, score AS similarity
MATCH (:User {uuid: $user_uuid})-[:OWNS_SOURCE]->(source:Source)
      -[:HAS_PAGE]->(:SourcePage)-[:CONTAINS_BLOCK]->(block)
WHERE block.content IS NOT NULL
WITH DISTINCT source, block, similarity
ORDER BY similarity DESC, block.uuid ASC
LIMIT $candidate_limit
CALL (source, block) {
    OPTIONAL MATCH (source)-[:HAS_PAGE]->(page:SourcePage)
          -[:CONTAINS_BLOCK]->(block)
    WITH DISTINCT page.index AS page_index
    ORDER BY page_index
    RETURN collect(page_index) AS page_indexes
}
CALL (block) {
    OPTIONAL MATCH (block)-[:FIRST_VISUAL_ASSET]->(first_asset:VisualAsset)
    OPTIONAL MATCH asset_path = (first_asset)
        -[:NEXT_VISUAL_ASSET*0..]->(asset:VisualAsset)
    WHERE (block)-[:CONTAINS_VISUAL_ASSET]->(asset)
    WITH asset, asset_path
    ORDER BY length(asset_path)
    RETURN [
        item IN collect(
            CASE WHEN asset IS NULL THEN NULL
            ELSE {uuid: asset.uuid, path: asset.path} END
        )
        WHERE item IS NOT NULL
    ] AS assets
}
RETURN {
    uuid: source.uuid,
    key: source.key,
    metadata: source.metadata,
    revision: coalesce(source.revision, 0)
} AS source,
{
    uuid: block.uuid,
    block_type: block.block_type,
    content: block.content,
    crop_path: block.crop_path,
    crop_bbox: block.crop_bbox,
    assets: assets
} AS block,
page_indexes,
similarity
"""

SEARCH_OWNED_SOURCE_STATEMENTS_VECTOR = """
CALL db.index.vector.queryNodes(
    'source_statement_embedding',
    $candidate_limit,
    $query_embedding
)
YIELD node AS statement, score AS similarity
MATCH (:User {uuid: $user_uuid})-[:OWNS_SOURCE]->(source:Source)
      -[:HAS_PAGE]->(:SourcePage)-[:CONTAINS_BLOCK]->(
          member:SourceBlock
      )-[:MEMBER_OF]->(statement)
WITH DISTINCT source, statement, similarity
ORDER BY similarity DESC, statement.uuid ASC
LIMIT $candidate_limit
CALL (source, statement) {
    MATCH (source)-[:FIRST_BLOCK]->(first:SourceBlock)
    MATCH path = (first)-[:NEXT_BLOCK*0..]->(block:SourceBlock)
    WHERE (source)-[:HAS_PAGE]->(:SourcePage)
              -[:CONTAINS_BLOCK]->(block)
      AND (block)-[:MEMBER_OF]->(statement)
    WITH block, length(path) AS block_order
    ORDER BY block_order
    RETURN collect(block.uuid) AS member_block_uuids
}
CALL (source, statement) {
    OPTIONAL MATCH (source)-[:HAS_PAGE]->(page:SourcePage)
          -[:CONTAINS_BLOCK]->(:SourceBlock)-[:MEMBER_OF]->(statement)
    WITH DISTINCT page.index AS page_index
    ORDER BY page_index
    RETURN collect(page_index) AS page_indexes
}
RETURN {
    uuid: source.uuid,
    key: source.key,
    metadata: source.metadata,
    revision: coalesce(source.revision, 0)
} AS source,
{
    uuid: statement.uuid,
    is_exercise: statement.is_exercise,
    content: statement.content,
    embedding: statement.embedding,
    member_block_uuids: member_block_uuids
} AS statement,
page_indexes,
similarity
"""

SEARCH_OWNED_SOURCE_PROCEDURES_VECTOR = """
CALL db.index.vector.queryNodes(
    'source_procedure_embedding',
    $candidate_limit,
    $query_embedding
)
YIELD node AS procedure, score AS similarity
MATCH (:User {uuid: $user_uuid})-[:OWNS_SOURCE]->(source:Source)
      -[:HAS_PAGE]->(:SourcePage)-[:CONTAINS_BLOCK]->(
          member:SourceBlock
      )-[:MEMBER_OF]->(procedure)
WITH DISTINCT source, procedure, similarity
ORDER BY similarity DESC, procedure.uuid ASC
LIMIT $candidate_limit
CALL (source, procedure) {
    MATCH (source)-[:FIRST_BLOCK]->(first:SourceBlock)
    MATCH path = (first)-[:NEXT_BLOCK*0..]->(block:SourceBlock)
    WHERE (source)-[:HAS_PAGE]->(:SourcePage)
              -[:CONTAINS_BLOCK]->(block)
      AND (block)-[:MEMBER_OF]->(procedure)
    WITH block, length(path) AS block_order
    ORDER BY block_order
    RETURN collect(block.uuid) AS member_block_uuids
}
CALL (source, procedure) {
    OPTIONAL MATCH (source)-[:HAS_PAGE]->(page:SourcePage)
          -[:CONTAINS_BLOCK]->(:SourceBlock)-[:MEMBER_OF]->(procedure)
    WITH DISTINCT page.index AS page_index
    ORDER BY page_index
    RETURN collect(page_index) AS page_indexes
}
RETURN {
    uuid: source.uuid,
    key: source.key,
    metadata: source.metadata,
    revision: coalesce(source.revision, 0)
} AS source,
{
    uuid: procedure.uuid,
    content: procedure.content,
    embedding: procedure.embedding,
    member_block_uuids: member_block_uuids
} AS procedure,
page_indexes,
similarity
"""

SEARCH_OWNED_SOURCE_STATEMENTS = """
MATCH (:User {uuid: $user_uuid})-[:OWNS_SOURCE]->(source:Source)
      -[:HAS_PAGE]->(:SourcePage)-[:CONTAINS_BLOCK]->(
          member:SourceBlock
      )-[:MEMBER_OF]->(statement:SourceStatement)
WHERE ($source_uuids IS NULL OR source.uuid IN $source_uuids)
  AND ($is_exercise IS NULL OR statement.is_exercise = $is_exercise)
  AND statement.embedding IS NOT NULL
WITH DISTINCT source, statement
WITH source, statement,
     vector.similarity.cosine(statement.embedding, $query_embedding)
         AS similarity
ORDER BY similarity DESC, statement.uuid ASC
LIMIT $candidate_limit
CALL (source, statement) {
    MATCH (source)-[:FIRST_BLOCK]->(first:SourceBlock)
    MATCH path = (first)-[:NEXT_BLOCK*0..]->(block:SourceBlock)
    WHERE (source)-[:HAS_PAGE]->(:SourcePage)
              -[:CONTAINS_BLOCK]->(block)
      AND (block)-[:MEMBER_OF]->(statement)
    WITH block, length(path) AS block_order
    ORDER BY block_order
    RETURN collect(block.uuid) AS member_block_uuids
}
CALL (source, statement) {
    OPTIONAL MATCH (source)-[:HAS_PAGE]->(page:SourcePage)
          -[:CONTAINS_BLOCK]->(:SourceBlock)-[:MEMBER_OF]->(statement)
    WITH DISTINCT page.index AS page_index
    ORDER BY page_index
    RETURN collect(page_index) AS page_indexes
}
RETURN {
    uuid: source.uuid,
    key: source.key,
    metadata: source.metadata,
    revision: coalesce(source.revision, 0)
} AS source,
{
    uuid: statement.uuid,
    is_exercise: statement.is_exercise,
    content: statement.content,
    embedding: statement.embedding,
    member_block_uuids: member_block_uuids
} AS statement,
page_indexes,
similarity
"""

SEARCH_OWNED_SOURCE_PROCEDURES = """
MATCH (:User {uuid: $user_uuid})-[:OWNS_SOURCE]->(source:Source)
      -[:HAS_PAGE]->(:SourcePage)-[:CONTAINS_BLOCK]->(
          member:SourceBlock
      )-[:MEMBER_OF]->(procedure:SourceProcedure)
WHERE ($source_uuids IS NULL OR source.uuid IN $source_uuids)
  AND procedure.embedding IS NOT NULL
WITH DISTINCT source, procedure
WITH source, procedure,
     vector.similarity.cosine(procedure.embedding, $query_embedding)
         AS similarity
ORDER BY similarity DESC, procedure.uuid ASC
LIMIT $candidate_limit
CALL (source, procedure) {
    MATCH (source)-[:FIRST_BLOCK]->(first:SourceBlock)
    MATCH path = (first)-[:NEXT_BLOCK*0..]->(block:SourceBlock)
    WHERE (source)-[:HAS_PAGE]->(:SourcePage)
              -[:CONTAINS_BLOCK]->(block)
      AND (block)-[:MEMBER_OF]->(procedure)
    WITH block, length(path) AS block_order
    ORDER BY block_order
    RETURN collect(block.uuid) AS member_block_uuids
}
CALL (source, procedure) {
    OPTIONAL MATCH (source)-[:HAS_PAGE]->(page:SourcePage)
          -[:CONTAINS_BLOCK]->(:SourceBlock)-[:MEMBER_OF]->(procedure)
    WITH DISTINCT page.index AS page_index
    ORDER BY page_index
    RETURN collect(page_index) AS page_indexes
}
RETURN {
    uuid: source.uuid,
    key: source.key,
    metadata: source.metadata,
    revision: coalesce(source.revision, 0)
} AS source,
{
    uuid: procedure.uuid,
    content: procedure.content,
    embedding: procedure.embedding,
    member_block_uuids: member_block_uuids
} AS procedure,
page_indexes,
similarity
"""

READ_OWNED_SOURCE_STATEMENTS = """
MATCH (:User {uuid: $user_uuid})-[:OWNS_SOURCE]->(source:Source)
      -[:HAS_PAGE]->(:SourcePage)-[:CONTAINS_BLOCK]->(
          member:SourceBlock
      )-[:MEMBER_OF]->(statement:SourceStatement)
WHERE statement.uuid IN $statement_uuids
WITH DISTINCT source, statement
CALL (source, statement) {
    MATCH (source)-[:FIRST_BLOCK]->(first:SourceBlock)
    MATCH path = (first)-[:NEXT_BLOCK*0..]->(block:SourceBlock)
    WHERE (source)-[:HAS_PAGE]->(:SourcePage)
              -[:CONTAINS_BLOCK]->(block)
      AND (block)-[:MEMBER_OF]->(statement)
    WITH block, length(path) AS block_order
    ORDER BY block_order
    RETURN collect(block.uuid) AS member_block_uuids
}
RETURN {
    uuid: source.uuid,
    key: source.key,
    metadata: source.metadata,
    revision: coalesce(source.revision, 0)
} AS source,
{
    uuid: statement.uuid,
    is_exercise: statement.is_exercise,
    content: statement.content,
    embedding: statement.embedding,
    member_block_uuids: member_block_uuids
} AS statement
ORDER BY statement.uuid
"""

READ_OWNED_SOURCE_PROCEDURES = """
MATCH (:User {uuid: $user_uuid})-[:OWNS_SOURCE]->(source:Source)
      -[:HAS_PAGE]->(:SourcePage)-[:CONTAINS_BLOCK]->(
          member:SourceBlock
      )-[:MEMBER_OF]->(procedure:SourceProcedure)
WHERE procedure.uuid IN $procedure_uuids
WITH DISTINCT source, procedure
CALL (source, procedure) {
    MATCH (source)-[:FIRST_BLOCK]->(first:SourceBlock)
    MATCH path = (first)-[:NEXT_BLOCK*0..]->(block:SourceBlock)
    WHERE (source)-[:HAS_PAGE]->(:SourcePage)
              -[:CONTAINS_BLOCK]->(block)
      AND (block)-[:MEMBER_OF]->(procedure)
    WITH block, length(path) AS block_order
    ORDER BY block_order
    RETURN collect(block.uuid) AS member_block_uuids
}
RETURN {
    uuid: source.uuid,
    key: source.key,
    metadata: source.metadata,
    revision: coalesce(source.revision, 0)
} AS source,
{
    uuid: procedure.uuid,
    content: procedure.content,
    embedding: procedure.embedding,
    member_block_uuids: member_block_uuids
} AS procedure
ORDER BY procedure.uuid
"""

READ_OWNED_SOURCE_PASSAGES = """
MATCH (:User {uuid: $user_uuid})-[:OWNS_SOURCE]->(source:Source)
      -[:HAS_PAGE]->(:SourcePage)-[:CONTAINS_BLOCK]->(
          selected_block:SourceBlock
      )
WHERE selected_block.uuid IN $block_uuids
WITH DISTINCT source, selected_block
WITH source,
     collect(selected_block) AS selected_blocks,
     collect(selected_block.uuid) AS selected_block_uuids
CALL (source, selected_blocks) {
    UNWIND CASE WHEN $expand_memberships
                THEN selected_blocks ELSE [] END AS selected_block
    OPTIONAL MATCH (selected_block)-[:MEMBER_OF]->(direct_group)
    WHERE direct_group:SourceStatement
       OR direct_group:SourceProcedure
       OR direct_group:Instruction
    WITH collect(DISTINCT direct_group) AS direct_groups
    WITH direct_groups,
         [group IN direct_groups
          WHERE group:SourceStatement] AS direct_statements
    OPTIONAL MATCH (instruction:Instruction)-[:GOVERNS]->(
        statement:SourceStatement
    )
    WHERE statement IN direct_statements
    WITH direct_groups, direct_statements,
         collect(DISTINCT instruction) AS governing_instructions
    WITH reduce(
        unique_groups = [],
        group IN direct_groups + governing_instructions |
        CASE WHEN group IN unique_groups
             THEN unique_groups
             ELSE unique_groups + group
        END
    ) AS group_nodes,
    [statement IN direct_statements | statement.uuid]
        AS direct_statement_uuids
    RETURN group_nodes, direct_statement_uuids
}
CALL (source, group_nodes, direct_statement_uuids) {
    UNWIND CASE WHEN size(group_nodes) = 0
                THEN [null] ELSE group_nodes END AS group_node
    OPTIONAL MATCH (source)-[:HAS_PAGE]->(:SourcePage)
          -[:CONTAINS_BLOCK]->(member_block:SourceBlock)
          -[:MEMBER_OF]->(group_node)
    WITH group_node, direct_statement_uuids,
         collect(DISTINCT member_block) AS member_blocks
    OPTIONAL MATCH (group_node)-[:GOVERNS]->(
        governed_statement:SourceStatement
    )
    WHERE group_node:Instruction
      AND governed_statement.uuid IN direct_statement_uuids
    WITH group_node, member_blocks,
         collect(DISTINCT governed_statement.uuid)
             AS governed_statement_uuids
    RETURN [
        group IN collect(
            CASE WHEN group_node IS NULL THEN NULL
            ELSE {
                uuid: group_node.uuid,
                kind: CASE
                    WHEN group_node:SourceStatement THEN 'statement'
                    WHEN group_node:SourceProcedure THEN 'procedure'
                    ELSE 'instruction'
                END,
                member_block_uuids: [
                    member IN member_blocks | member.uuid
                ],
                member_blocks: member_blocks,
                governed_statement_uuids: governed_statement_uuids
            } END
        )
        WHERE group IS NOT NULL
    ] AS groups
}
WITH source, selected_blocks, selected_block_uuids, groups,
     reduce(
         core_blocks = selected_blocks,
         group IN groups | core_blocks + group.member_blocks
     ) AS core_blocks
CALL (source, core_blocks) {
    UNWIND core_blocks AS core_block
    OPTIONAL MATCH (neighbor:SourceBlock)
        -[:NEXT_BLOCK*0..{before_hops}]->(core_block)
    WHERE neighbor.uuid <> core_block.uuid
      AND EXISTS {
          MATCH (source)-[:HAS_PAGE]->(:SourcePage)
                -[:CONTAINS_BLOCK]->(neighbor)
      }
    RETURN collect(DISTINCT neighbor) AS preceding_blocks
}
CALL (source, core_blocks) {
    UNWIND core_blocks AS core_block
    OPTIONAL MATCH (core_block)
        -[:NEXT_BLOCK*0..{after_hops}]->(neighbor:SourceBlock)
    WHERE neighbor.uuid <> core_block.uuid
      AND EXISTS {
          MATCH (source)-[:HAS_PAGE]->(:SourcePage)
                -[:CONTAINS_BLOCK]->(neighbor)
      }
    RETURN collect(DISTINCT neighbor) AS following_blocks
}
WITH source, selected_blocks, selected_block_uuids, groups,
     [block IN core_blocks + preceding_blocks + following_blocks
      WHERE block IS NOT NULL] AS candidate_blocks
WITH source, selected_blocks, selected_block_uuids, groups,
     reduce(
         unique_blocks = [],
         block IN candidate_blocks |
         CASE WHEN block IN unique_blocks
              THEN unique_blocks
              ELSE unique_blocks + block
         END
     ) AS result_blocks
MATCH (source)-[:FIRST_BLOCK]->(first_block:SourceBlock)
MATCH order_path = (first_block)-[:NEXT_BLOCK*0..]->(block:SourceBlock)
WHERE block IN result_blocks
WITH source, selected_block_uuids, groups, block,
     length(order_path) AS block_order
ORDER BY block_order
CALL (source, block) {
    OPTIONAL MATCH (source)-[:HAS_PAGE]->(page:SourcePage)
          -[:CONTAINS_BLOCK]->(block)
    WITH DISTINCT page.index AS page_index
    ORDER BY page_index
    RETURN collect(page_index) AS page_indexes
}
CALL (block) {
    OPTIONAL MATCH (block)-[:FIRST_VISUAL_ASSET]->(first_asset:VisualAsset)
    OPTIONAL MATCH asset_path = (first_asset)
        -[:NEXT_VISUAL_ASSET*0..]->(asset:VisualAsset)
    WHERE (block)-[:CONTAINS_VISUAL_ASSET]->(asset)
    WITH asset, asset_path
    ORDER BY length(asset_path)
    RETURN [
        item IN collect(
            CASE WHEN asset IS NULL THEN NULL
            ELSE {uuid: asset.uuid, path: asset.path} END
        )
        WHERE item IS NOT NULL
    ] AS assets
}
WITH source, selected_block_uuids, groups, block_order,
{
    uuid: block.uuid,
    block_type: block.block_type,
    content: block.content,
    crop_path: block.crop_path,
    crop_bbox: block.crop_bbox,
    assets: assets
} AS block_data,
page_indexes
ORDER BY block_order
WITH source, selected_block_uuids, groups,
     collect({
         block: block_data,
         page_indexes: page_indexes
     }) AS blocks
RETURN {
    uuid: source.uuid,
    key: source.key,
    metadata: source.metadata,
    revision: coalesce(source.revision, 0)
} AS source,
selected_block_uuids,
blocks,
[
    group IN groups | {
        uuid: group.uuid,
        kind: group.kind,
        member_block_uuids: group.member_block_uuids,
        governed_statement_uuids: group.governed_statement_uuids
    }
] AS groups
"""


def read_source_passages_query(*, before: int, after: int) -> str:
    """Build bounded source-chain traversal for whole-block neighbors."""
    return READ_OWNED_SOURCE_PASSAGES.replace(
        '{before_hops}', str(before)
    ).replace('{after_hops}', str(after))


READ_OWNED_SOURCE_ASSETS = """
UNWIND $asset_uuids AS asset_uuid
MATCH (:User {uuid: $user_uuid})-[:OWNS_SOURCE]->(:Source)
      -[:HAS_PAGE]->(:SourcePage)-[:CONTAINS_BLOCK]->(:SourceBlock)
      -[:CONTAINS_VISUAL_ASSET]->(asset:VisualAsset {uuid: asset_uuid})
WITH DISTINCT asset.uuid AS uuid, asset.path AS path
RETURN uuid, path
"""

READ_OWNED_SOURCE_ASSETS_AT_REVISIONS = """
UNWIND $image_references AS requested
MATCH (:User {uuid: $user_uuid})-[:OWNS_SOURCE]->
      (source:Source {uuid: requested.source_uuid})
      -[:HAS_PAGE]->(:SourcePage)-[:CONTAINS_BLOCK]->(:SourceBlock)
      -[:CONTAINS_VISUAL_ASSET]->
      (asset:VisualAsset {uuid: requested.asset_uuid})
WHERE coalesce(source.revision, 0) = requested.revision
WITH DISTINCT requested.position AS position, source, asset
ORDER BY position
RETURN source.uuid AS source_uuid,
       coalesce(source.revision, 0) AS revision,
       asset.uuid AS uuid,
       asset.path AS path
"""
