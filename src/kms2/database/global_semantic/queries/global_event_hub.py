"""Cypher statements for global event hubs."""

READ_GLOBAL_EVENT_HUB_CANDIDATES = """
MATCH (query:SourceEventHub)
WHERE query.embedding IS NOT NULL
CALL db.index.vector.queryNodes(
    'source_event_hub_embedding',
    $candidate_limit,
    query.embedding
)
YIELD node AS candidate, score
WHERE candidate.uuid <> query.uuid
  AND candidate.source_uuid <> query.source_uuid
  AND score >= $minimum_similarity
WITH CASE WHEN query.uuid < candidate.uuid THEN query ELSE candidate END AS left,
     CASE WHEN query.uuid < candidate.uuid THEN candidate ELSE query END AS right,
     score
WITH left,
     right,
     max(score) AS score
RETURN left.uuid AS left_uuid,
       left.name AS left_name,
       left.description AS left_description,
       right.uuid AS right_uuid,
       right.name AS right_name,
       right.description AS right_description,
       score
ORDER BY left_uuid, right_uuid
"""

REPLACE_GLOBAL_EVENT_HUB_ACCEPTED_EDGES = """
MATCH (left:SourceEventHub)-[old_similarity:SIMILAR_TO]-
      (right:SourceEventHub)
DELETE old_similarity
WITH count(*) AS deleted
UNWIND $pairs AS pair
MATCH (left:SourceEventHub {uuid: pair.left_uuid})
MATCH (right:SourceEventHub {uuid: pair.right_uuid})
MERGE (left)-[similarity:SIMILAR_TO]->(right)
SET similarity.score = pair.score
RETURN count(*) AS materialized
"""

DROP_GLOBAL_EVENT_HUB_GRAPH = """
CALL gds.graph.drop(
    $graph_name,
    false
)
YIELD graphName
RETURN graphName
"""

DETECT_GLOBAL_EVENT_HUB_COMMUNITIES = """
CALL gds.graph.project(
    $graph_name,
    'SourceEventHub',
    {
        SIMILAR_TO: {
            orientation: 'UNDIRECTED',
            properties: 'score'
        }
    }
)
YIELD graphName
CALL gds.sllpa.stream(
    $graph_name,
    {
        relationshipWeightProperty: 'score',
        maxIterations: $max_iterations,
        minAssociationStrength: $min_association_strength
    }
)
YIELD nodeId, values
WITH gds.util.asNode(nodeId) AS node,
     values.communityIds AS communityIds
UNWIND communityIds AS community_id
RETURN community_id,
       node.uuid AS uuid,
       node.name AS name,
       node.description AS description
ORDER BY community_id, uuid
"""

REPLACE_GLOBAL_EVENT_HUBS = """
CALL () {
    MATCH (old_hub:GlobalEventHub)
    DETACH DELETE old_hub
    RETURN count(*) AS deleted
}
UNWIND $hubs AS hub_data
CREATE (hub:GlobalEventHub {
    uuid: hub_data.uuid,
    name: hub_data.name,
    aliases: hub_data.aliases,
    description: hub_data.description,
    embedding: hub_data.embedding
})
WITH hub,
     hub_data
UNWIND hub_data.member_uuids AS member_uuid
MATCH (member:SourceEventHub {uuid: member_uuid})
CREATE (member)-[:IN_GLOBAL_HUB]->(hub)
RETURN count(DISTINCT hub) AS persisted
"""
