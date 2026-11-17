"""Cypher statements for owner-scoped v5 source retrieval."""

BLOCK_PROJECTION = """{
    uuid: block.uuid, created_at: block.created_at,
    updated_at: block.updated_at, invalidated_at: block.invalidated_at,
    block_type: block.block_type, content: block.content,
    crop_path: block.crop_path, crop_sha256: block.crop_sha256,
    crop_bbox: block.crop_bbox, image_path: block.image_path,
    image_sha256: block.image_sha256
}"""

SOURCE_PROJECTION = """{
    uuid: source.uuid, key: source.key, metadata: source.metadata,
    identity_schema: source.identity_schema, created_at: source.created_at,
    updated_at: source.updated_at, invalidated_at: source.invalidated_at,
    revision: coalesce(source.revision, 0)
}"""

SEARCH_OWNED_SOURCE_BLOCKS = f"""
MATCH (:User {{uuid: $user_uuid}})-[:OWNS_SOURCE]->(source:Source)
      -[:HAS_UNIT]->(unit:SourceUnit)-[:HAS_BLOCK]->(block:SourceBlock)
WHERE source.identity_schema = $identity_schema
  AND ($source_uuids IS NULL OR source.uuid IN $source_uuids)
  AND block.embedding IS NOT NULL
WITH source, unit, block,
     vector.similarity.cosine(block.embedding, $query_embedding) AS similarity
ORDER BY similarity DESC, block.uuid ASC
LIMIT $candidate_limit
RETURN {SOURCE_PROJECTION} AS source, unit.uuid AS unit_uuid,
       {BLOCK_PROJECTION} AS block, similarity
"""

SEARCH_OWNED_SOURCE_BLOCKS_VECTOR = f"""
CALL db.index.vector.queryNodes('source_block_embedding', $candidate_limit,
                                $query_embedding)
YIELD node AS block, score AS similarity
MATCH (:User {{uuid: $user_uuid}})-[:OWNS_SOURCE]->(source:Source)
      -[:HAS_UNIT]->(unit:SourceUnit)-[:HAS_BLOCK]->(block)
WHERE source.identity_schema = $identity_schema
WITH DISTINCT source, unit, block, similarity
ORDER BY similarity DESC, block.uuid ASC
LIMIT $candidate_limit
RETURN {SOURCE_PROJECTION} AS source, unit.uuid AS unit_uuid,
       {BLOCK_PROJECTION} AS block, similarity
"""

_UNIT_SEARCH = """
MATCH (:User {uuid: $user_uuid})-[:OWNS_SOURCE]->(source:Source)
      -[:HAS_UNIT]->(unit:SourceUnit)
WHERE source.identity_schema = $identity_schema
  AND ($source_uuids IS NULL OR source.uuid IN $source_uuids)
  AND ($kind IS NULL OR unit.kind = $kind)
  AND unit.embedding IS NOT NULL
WITH source, unit,
     vector.similarity.cosine(unit.embedding, $query_embedding) AS similarity
ORDER BY similarity DESC, unit.uuid ASC
LIMIT $candidate_limit
"""

_UNIT_VECTOR_SEARCH = """
CALL db.index.vector.queryNodes('source_unit_embedding', $candidate_limit,
                                $query_embedding)
YIELD node AS unit, score AS similarity
MATCH (:User {uuid: $user_uuid})-[:OWNS_SOURCE]->(source:Source)
      -[:HAS_UNIT]->(unit)
WHERE source.identity_schema = $identity_schema
WITH DISTINCT source, unit, similarity
ORDER BY similarity DESC, unit.uuid ASC
LIMIT $candidate_limit
"""

_UNIT_BLOCKS = """
CALL (unit) {
    MATCH (unit)-[:FIRST_BLOCK]->(first:SourceBlock)
    MATCH chain=(first)-[:NEXT_BLOCK*0..]->(block:SourceBlock)
    WHERE all(member IN nodes(chain) WHERE (unit)-[:HAS_BLOCK]->(member))
    WITH block, length(chain) AS order_index
    ORDER BY order_index
    RETURN collect({
        uuid: block.uuid, created_at: block.created_at,
        updated_at: block.updated_at, invalidated_at: block.invalidated_at,
        block_type: block.block_type, content: block.content,
        crop_path: block.crop_path, crop_sha256: block.crop_sha256,
        crop_bbox: block.crop_bbox, image_path: block.image_path,
        image_sha256: block.image_sha256
    }) AS blocks
}
"""


SEARCH_OWNED_SOURCE_UNITS = f"""
{_UNIT_SEARCH}
{_UNIT_BLOCKS}
RETURN {SOURCE_PROJECTION} AS source,
       unit{{.*, invalidated_at: unit.invalidated_at, blocks: blocks}} AS unit, similarity
"""

SEARCH_OWNED_SOURCE_UNITS_VECTOR = f"""
{_UNIT_VECTOR_SEARCH}
{_UNIT_BLOCKS}
RETURN {SOURCE_PROJECTION} AS source,
       unit{{.*, invalidated_at: unit.invalidated_at, blocks: blocks}} AS unit, similarity
"""

READ_OWNED_SOURCE_BLOCKS = f"""
UNWIND $block_uuids AS block_uuid
MATCH (:User {{uuid: $user_uuid}})-[:OWNS_SOURCE]->(source:Source)
      -[:HAS_UNIT]->(unit:SourceUnit)-[:HAS_BLOCK]->(block:SourceBlock {{uuid: block_uuid}})
WHERE source.identity_schema = $identity_schema
RETURN {SOURCE_PROJECTION} AS source, unit.uuid AS unit_uuid,
       {BLOCK_PROJECTION} AS block
"""


def read_source_units_query(*, before: int, after: int) -> str:
    """Build bounded same-source NEXT_UNIT neighborhoods for requested units."""
    return f"""
UNWIND $unit_uuids AS unit_uuid
MATCH (:User {{uuid: $user_uuid}})-[:OWNS_SOURCE]->(source:Source)
      -[:HAS_UNIT]->(requested:SourceUnit {{uuid: unit_uuid}})
WHERE source.identity_schema = $identity_schema
WITH DISTINCT source, requested
CALL (source, requested) {{
    MATCH chain=(previous:SourceUnit)-[:NEXT_UNIT*0..{before}]->(requested)
    WHERE all(member IN nodes(chain) WHERE (source)-[:HAS_UNIT]->(member))
    RETURN collect(DISTINCT previous) AS previous_units
}}
CALL (source, requested) {{
    MATCH chain=(requested)-[:NEXT_UNIT*0..{after}]->(following:SourceUnit)
    WHERE all(member IN nodes(chain) WHERE (source)-[:HAS_UNIT]->(member))
    RETURN collect(DISTINCT following) AS following_units
}}
WITH source, previous_units + following_units AS neighborhood
UNWIND neighborhood AS unit
WITH DISTINCT source, unit
MATCH (source)-[:FIRST_UNIT]->(first:SourceUnit)
MATCH chain=(first)-[:NEXT_UNIT*0..]->(unit)
WHERE all(member IN nodes(chain) WHERE (source)-[:HAS_UNIT]->(member))
WITH source, unit, length(chain) AS order_index
ORDER BY source.uuid, order_index
{_UNIT_BLOCKS}
WITH source, unit, blocks, order_index
ORDER BY source.uuid, order_index
WITH source, collect(unit{{.*, invalidated_at: unit.invalidated_at, blocks: blocks, order_index: order_index}}) AS units
RETURN {SOURCE_PROJECTION} AS source, units
"""


READ_OWNED_SOURCE_IMAGES = """
UNWIND $block_uuids AS block_uuid
MATCH (:User {uuid: $user_uuid})-[:OWNS_SOURCE]->(source:Source)
      -[:HAS_UNIT]->(:SourceUnit)-[:HAS_BLOCK]->(block:SourceBlock {uuid: block_uuid})
WHERE source.identity_schema = $identity_schema
RETURN block.uuid AS block_uuid, source.identity_schema AS identity_schema,
       block.block_type AS block_type, block.image_path AS image_path,
       block.image_sha256 AS image_sha256
"""
