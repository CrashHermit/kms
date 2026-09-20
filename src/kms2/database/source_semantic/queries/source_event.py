"""Cypher statements for source event."""

READ_SOURCE_EVENTS = """
MATCH (source_event:SourceEvent {source_uuid: $source_uuid})
RETURN source_event.uuid AS uuid,
       source_event.source_uuid AS source_uuid,
       source_event.source_block_uuid AS source_block_uuid,
       source_event.name AS name
ORDER BY source_event.source_block_uuid, source_event.uuid
"""

UPDATE_SOURCE_EVENT_DESCRIPTION = """
UNWIND $rows AS row
MATCH (source_event:SourceEvent {
    uuid: row.uuid,
    source_uuid: $source_uuid
})
SET source_event.description = row.description,
    source_event.embedding = row.embedding
RETURN count(source_event) AS updated
"""

FIND_SIMILAR_SOURCE_EVENTS = """
MATCH (query:SourceEvent {uuid: $query_uuid})
CALL db.index.vector.queryNodes(
    'source_event_embedding',
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

READ_SOURCE_EVENT_HUB_CANDIDATES = """
MATCH (query:SourceEvent {source_uuid: $source_uuid})
WHERE query.embedding IS NOT NULL
CALL db.index.vector.queryNodes('source_event_embedding', $candidate_limit, query.embedding)
YIELD node AS candidate, score
WHERE candidate.source_uuid = $source_uuid AND candidate.uuid <> query.uuid AND score >= $minimum_similarity
WITH CASE WHEN query.uuid < candidate.uuid THEN query ELSE candidate END AS left,
     CASE WHEN query.uuid < candidate.uuid THEN candidate ELSE query END AS right, score
WITH left, right, max(score) AS score
RETURN left.uuid AS left_uuid, left.name AS left_name, left.description AS left_description,
       right.uuid AS right_uuid, right.name AS right_name, right.description AS right_description, score
ORDER BY left_uuid, right_uuid
"""

REPLACE_SOURCE_EVENT_ACCEPTED_EDGES = """
MATCH (left:SourceEvent {source_uuid: $source_uuid})-[old_similarity:SIMILAR_TO]-(right:SourceEvent {source_uuid: $source_uuid})
WHERE left.uuid < right.uuid DELETE old_similarity
WITH count(*) AS deleted
UNWIND $pairs AS pair
MATCH (left:SourceEvent {uuid: pair.left_uuid, source_uuid: $source_uuid})
MATCH (right:SourceEvent {uuid: pair.right_uuid, source_uuid: $source_uuid})
MERGE (left)-[similarity:SIMILAR_TO]->(right) SET similarity.score = pair.score
RETURN count(*) AS materialized
"""

DROP_SOURCE_EVENT_HUB_GRAPH = """CALL gds.graph.drop($graph_name, false) YIELD graphName RETURN graphName"""

DETECT_SOURCE_EVENT_COMMUNITIES = """
CALL gds.graph.project($graph_name, 'SourceEvent', {SIMILAR_TO: {orientation: 'UNDIRECTED', properties: 'score'}})
YIELD graphName
CALL gds.sllpa.stream($graph_name, {relationshipWeightProperty: 'score', maxIterations: $max_iterations, minAssociationStrength: $min_association_strength})
YIELD nodeId, values
WITH gds.util.asNode(nodeId) AS node, values.communityIds AS communityIds
UNWIND communityIds AS community_id
MATCH (node:SourceEvent {source_uuid: $source_uuid})
RETURN community_id, node.uuid AS uuid, node.name AS name, node.description AS description
ORDER BY community_id, uuid
"""

REPLACE_SOURCE_EVENT_HUBS = """
CALL { MATCH (old_hub:SourceEventHub {source_uuid: $source_uuid}) DETACH DELETE old_hub RETURN count(*) AS deleted }
UNWIND $hubs AS hub_data
CREATE (hub:SourceEventHub {uuid: hub_data.uuid, source_uuid: $source_uuid, name: hub_data.name, aliases: hub_data.aliases, description: hub_data.description, embedding: hub_data.embedding})
WITH hub, hub_data
UNWIND hub_data.member_uuids AS member_uuid
MATCH (member:SourceEvent {uuid: member_uuid, source_uuid: $source_uuid})
CREATE (member)-[:IN_SOURCE_HUB]->(hub)
RETURN count(DISTINCT hub) AS persisted
"""
