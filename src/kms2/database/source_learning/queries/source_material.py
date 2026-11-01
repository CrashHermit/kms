"""Cypher statements for loading source-material flashcard requests."""

READ_SOURCE_ATOMIC_FLASHCARD_REQUESTS = """
MATCH (source:Source {uuid: $source_uuid})
      -[:HAS_PAGE]->(page:SourcePage)-[:CONTAINS_BLOCK]->(owner:SourceBlock)
WHERE owner.content IS NOT NULL AND trim(owner.content) <> ''
MATCH (source)-[:FIRST_BLOCK]->(first_block:SourceBlock)
MATCH path = (first_block)-[:NEXT_BLOCK*0..]->(owner)
RETURN owner.uuid AS source_block_uuid,
       page.index AS source_page_index,
       owner.block_type AS block_type,
       owner.content AS content,
       length(path) AS owner_position
ORDER BY owner_position, source_block_uuid
"""
