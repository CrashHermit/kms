"""Cypher statements for source entity."""

READ_SOURCE_ENTITIES = """
MATCH (source_entity:SourceEntity {source_uuid: $source_uuid})
RETURN source_entity.uuid AS uuid,
       source_entity.source_uuid AS source_uuid,
       source_entity.source_block_uuid AS source_block_uuid,
       source_entity.name AS name
ORDER BY source_entity.source_block_uuid, source_entity.uuid
"""

UPDATE_SOURCE_ENTITY_DESCRIPTION = """
UNWIND $rows AS row
MATCH (source_entity:SourceEntity {
    uuid: row.uuid,
    source_uuid: $source_uuid
})
SET source_entity.description = row.description,
    source_entity.embedding = row.embedding
RETURN count(source_entity) AS updated
"""

FIND_SIMILAR_SOURCE_ENTITIES = """
MATCH (query:SourceEntity {uuid: $query_uuid})
CALL db.index.vector.queryNodes(
    'source_entity_embedding',
    $candidate_limit,
    query.embedding
)
YIELD node, score
WHERE node.uuid <> query.uuid
RETURN node.uuid AS uuid,
       node.source_uuid AS source_uuid,
       node.source_block_uuid AS source_block_uuid,
       node.name AS name,
       node.description AS description,
       score
ORDER BY score DESC, uuid ASC
LIMIT $top_k
"""

READ_SOURCE_ENTITY_HUB_CANDIDATES = """
MATCH (query:SourceEntity {source_uuid: $source_uuid})
WHERE query.embedding IS NOT NULL
CALL db.index.vector.queryNodes('source_entity_embedding', $candidate_limit, query.embedding)
YIELD node AS candidate, score
WHERE candidate.source_uuid = $source_uuid AND candidate.uuid <> query.uuid AND score >= $minimum_similarity
WITH CASE WHEN query.uuid < candidate.uuid THEN query ELSE candidate END AS left,
     CASE WHEN query.uuid < candidate.uuid THEN candidate ELSE query END AS right, score
WITH left, right, max(score) AS score
RETURN left.uuid AS left_uuid, left.name AS left_name, left.description AS left_description,
       right.uuid AS right_uuid, right.name AS right_name, right.description AS right_description, score
ORDER BY left_uuid, right_uuid
"""

REPLACE_SOURCE_ENTITY_ACCEPTED_EDGES = """
MATCH (left:SourceEntity {source_uuid: $source_uuid})-[old_similarity:SIMILAR_TO]-(right:SourceEntity {source_uuid: $source_uuid})
WHERE left.uuid < right.uuid DELETE old_similarity
WITH count(*) AS deleted
UNWIND $pairs AS pair
MATCH (left:SourceEntity {uuid: pair.left_uuid, source_uuid: $source_uuid})
MATCH (right:SourceEntity {uuid: pair.right_uuid, source_uuid: $source_uuid})
MERGE (left)-[similarity:SIMILAR_TO]->(right) SET similarity.score = pair.score
RETURN count(*) AS materialized
"""

DROP_SOURCE_ENTITY_HUB_GRAPH = """CALL gds.graph.drop($graph_name, false) YIELD graphName RETURN graphName"""

DETECT_SOURCE_ENTITY_COMMUNITIES = """
CALL gds.graph.project($graph_name, 'SourceEntity', {SIMILAR_TO: {orientation: 'UNDIRECTED', properties: 'score'}})
YIELD graphName
CALL gds.sllpa.stream($graph_name, {relationshipWeightProperty: 'score', maxIterations: $max_iterations, minAssociationStrength: $min_association_strength})
YIELD nodeId, values
WITH gds.util.asNode(nodeId) AS node, values.communityIds AS communityIds
UNWIND communityIds AS community_id
MATCH (node:SourceEntity {source_uuid: $source_uuid})
RETURN community_id, node.uuid AS uuid, node.name AS name, node.description AS description
ORDER BY community_id, uuid
"""

REPLACE_SOURCE_ENTITY_HUBS = """
CALL { MATCH (old_hub:SourceEntityHub {source_uuid: $source_uuid}) DETACH DELETE old_hub RETURN count(*) AS deleted }
UNWIND $hubs AS hub_data
CREATE (hub:SourceEntityHub {uuid: hub_data.uuid, source_uuid: $source_uuid, canonical_name: hub_data.canonical_name, aliases: hub_data.aliases, description: hub_data.description, embedding: hub_data.embedding})
WITH hub, hub_data
UNWIND hub_data.member_uuids AS member_uuid
MATCH (member:SourceEntity {uuid: member_uuid, source_uuid: $source_uuid})
CREATE (member)-[:IN_SOURCE_HUB]->(hub)
RETURN count(DISTINCT hub) AS persisted
"""
