"""Cypher statements for source entity."""

REPLACE_SOURCE_ENTITIES = """
MATCH (source:Source {uuid: $source_uuid})
CALL (source) {
    OPTIONAL MATCH (source)-[:HAS_PAGE]->(:SourcePage)
        -[:CONTAINS_BLOCK]->(:SourceBlock)-[:MEMBER_OF]->(:SourceEntity)
        -[:IN_SOURCE_HUB]->(hub:SourceEntityHub)
    WITH collect(DISTINCT hub) AS hubs
    FOREACH (hub IN hubs | DETACH DELETE hub)
    RETURN count(*) AS old_hubs_deleted
}
CALL (source) {
    OPTIONAL MATCH (source)-[:HAS_PAGE]->(:SourcePage)
        -[:CONTAINS_BLOCK]->(:SourceBlock)-[:MEMBER_OF]->(old:SourceEntity)
    WITH collect(DISTINCT old) AS old_entities
    FOREACH (entity IN old_entities | DETACH DELETE entity)
    RETURN count(*) AS old_occurrences_deleted
}
CALL (source) {
    UNWIND $entities AS row
    CREATE (entity:SourceEntity {
        uuid: row.uuid,
        name: row.name
    })
    WITH entity, row
    MATCH (source)-[:HAS_PAGE]->(:SourcePage)
        -[:CONTAINS_BLOCK]->(target_block:SourceBlock {
            uuid: row.target_block_uuid
        })
    CREATE (target_block)-[:MEMBER_OF]->(entity)
    RETURN count(DISTINCT entity) AS created
}
RETURN source.uuid AS uuid
"""

READ_SOURCE_ENTITIES = """
MATCH (source:Source {uuid: $source_uuid})-[:FIRST_BLOCK]->(
    first_block:SourceBlock
)
MATCH path = (first_block)-[:NEXT_BLOCK*0..]->(target_block:SourceBlock)
MATCH (target_block)-[:MEMBER_OF]->(source_entity:SourceEntity)
RETURN source_entity.uuid AS uuid,
       source.uuid AS source_uuid,
       source_entity.name AS name,
       source_entity.description AS description,
       target_block.uuid AS source_block_uuid,
       [{uuid: target_block.uuid,
         block_type: target_block.block_type,
         content: target_block.content}] AS target_blocks
ORDER BY length(path), source_entity.uuid
"""


UPDATE_SOURCE_ENTITY_DESCRIPTION = """
UNWIND $rows AS row
MATCH (source:Source {uuid: $source_uuid})-[:HAS_PAGE]->(:SourcePage)
      -[:CONTAINS_BLOCK]->(:SourceBlock)-[:MEMBER_OF]->(
          source_entity:SourceEntity {uuid: row.uuid}
      )
SET source_entity.description = row.description,
    source_entity.embedding = row.embedding
RETURN count(DISTINCT source_entity) AS updated
"""


READ_SOURCE_ENTITY_HUB_CANDIDATES = """
MATCH (source:Source {uuid: $source_uuid})
UNWIND $groups AS query_group
CALL (source, query_group) {
    UNWIND $groups AS candidate_group
    WITH source, query_group, candidate_group
    WHERE candidate_group.group_uuid <> query_group.group_uuid
    MATCH (source)-[:HAS_PAGE]->(:SourcePage)-[:CONTAINS_BLOCK]->(
        :SourceBlock
    )-[:MEMBER_OF]->(query_member:SourceEntity)
    WHERE query_member.uuid IN query_group.member_uuids
          AND query_member.embedding IS NOT NULL
    MATCH (source)-[:HAS_PAGE]->(:SourcePage)-[:CONTAINS_BLOCK]->(
        :SourceBlock
    )-[:MEMBER_OF]->(candidate_member:SourceEntity)
    WHERE candidate_member.uuid IN candidate_group.member_uuids
          AND candidate_member.embedding IS NOT NULL
    WITH query_group, candidate_group, query_member, candidate_member,
         vector.similarity.cosine(
             query_member.embedding,
             candidate_member.embedding
         ) AS score
    WHERE score >= $minimum_similarity
    ORDER BY score DESC, query_member.uuid, candidate_member.uuid
    WITH query_group, candidate_group,
         collect({
             query_member: query_member,
             candidate_member: candidate_member,
             score: score
         })[0] AS best,
         max(score) AS score
    ORDER BY score DESC, candidate_group.group_uuid
    LIMIT $candidate_limit
    RETURN candidate_group, best, score
}
WITH CASE WHEN query_group.group_uuid < candidate_group.group_uuid
          THEN query_group.group_uuid ELSE candidate_group.group_uuid
     END AS left_group_uuid,
     CASE WHEN query_group.group_uuid < candidate_group.group_uuid
          THEN candidate_group.group_uuid ELSE query_group.group_uuid
     END AS right_group_uuid,
     CASE WHEN query_group.group_uuid < candidate_group.group_uuid
          THEN best.query_member ELSE best.candidate_member
     END AS left_member,
     CASE WHEN query_group.group_uuid < candidate_group.group_uuid
          THEN best.candidate_member ELSE best.query_member
     END AS right_member,
     score
ORDER BY score DESC, left_member.uuid, right_member.uuid
WITH left_group_uuid, right_group_uuid,
     collect({
         left: left_member,
         right: right_member,
         score: score
     })[0] AS best,
     max(score) AS score
RETURN left_group_uuid,
       right_group_uuid,
       best.left.uuid AS left_uuid,
       best.right.uuid AS right_uuid,
       score
ORDER BY score DESC, left_group_uuid, right_group_uuid
"""


SEARCH_SOURCE_ENTITY_HUBS = """
MATCH (source:Source {uuid: $source_uuid})-[:HAS_PAGE]->(:SourcePage)
      -[:CONTAINS_BLOCK]->(:SourceBlock)-[:MEMBER_OF]->(member:SourceEntity)
      -[:IN_SOURCE_HUB]->(hub:SourceEntityHub)
WHERE member.embedding IS NOT NULL
WITH source, hub, member,
     vector.similarity.cosine(member.embedding, $query_embedding) AS score
ORDER BY hub.uuid, score DESC, member.uuid
WITH source, hub, collect({
    uuid: member.uuid, name: member.name, description: member.description,
    score: score
})[0] AS best
MATCH (source)-[:HAS_PAGE]->(:SourcePage)-[:CONTAINS_BLOCK]->(
    target_block:SourceBlock
)-[:MEMBER_OF]->(:SourceEntity {uuid: best.uuid})
MATCH (source)-[:FIRST_BLOCK]->(first:SourceBlock)
MATCH path = (first)-[:NEXT_BLOCK*0..]->(target_block)
WITH source, hub, best, target_block, path
ORDER BY length(path)
WITH source, hub, best, collect(target_block.uuid) AS target_block_uuids
RETURN hub.uuid AS uuid, source.uuid AS source_uuid,
       best.uuid AS member_uuid, target_block_uuids,
       best.name AS name, best.description AS description, best.score AS score
ORDER BY score DESC, uuid
LIMIT $top_k
"""

REPLACE_SOURCE_ENTITY_HUBS = """
MATCH (source:Source {uuid: $source_uuid})
CALL (source) {
    OPTIONAL MATCH (source)-[:HAS_PAGE]->(:SourcePage)
        -[:CONTAINS_BLOCK]->(:SourceBlock)-[:MEMBER_OF]->(:SourceEntity)
        -[:IN_SOURCE_HUB]->(old_hub:SourceEntityHub)
    WITH collect(DISTINCT old_hub) AS old_hubs
    FOREACH (hub IN old_hubs | DETACH DELETE hub)
    RETURN count(*) AS deleted
}
CALL (source) {
    UNWIND $hubs AS hub_data
    CREATE (hub:SourceEntityHub {uuid: hub_data.uuid})
    WITH source, hub, hub_data
    UNWIND hub_data.member_uuids AS member_uuid
    MATCH (source)-[:HAS_PAGE]->(:SourcePage)-[:CONTAINS_BLOCK]->(:SourceBlock)
          -[:MEMBER_OF]->(member:SourceEntity {uuid: member_uuid})
    CREATE (member)-[:IN_SOURCE_HUB]->(hub)
    RETURN count(DISTINCT hub) AS persisted
}
RETURN source.uuid AS uuid, persisted
"""
