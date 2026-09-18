"""Cypher statements for ordered source-block access and search."""

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
