"""Cypher statements for global entity hubs."""

READ_GLOBAL_ENTITY_HUB_CANDIDATES = """
MATCH (query:SourceEntityHub)
WHERE query.embedding IS NOT NULL
CALL db.index.vector.queryNodes(
    'source_entity_hub_embedding',
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
WITH left, right, max(score) AS score
RETURN left.uuid AS left_uuid,
       left.canonical_name AS left_canonical_name,
       left.description AS left_description,
       right.uuid AS right_uuid,
       right.canonical_name AS right_canonical_name,
       right.description AS right_description,
       score
ORDER BY left_uuid, right_uuid
"""

REPLACE_GLOBAL_ENTITY_HUB_ACCEPTED_EDGES = """
MATCH (left:SourceEntityHub)-[old_similarity:SIMILAR_TO]-
      (right:SourceEntityHub)
DELETE old_similarity
WITH count(*) AS deleted
UNWIND $pairs AS pair
MATCH (left:SourceEntityHub {uuid: pair.left_uuid})
MATCH (right:SourceEntityHub {uuid: pair.right_uuid})
MERGE (left)-[similarity:SIMILAR_TO]->(right)
SET similarity.score = pair.score
RETURN count(*) AS materialized
"""

DROP_GLOBAL_ENTITY_HUB_GRAPH = """
CALL gds.graph.drop($graph_name, false) YIELD graphName RETURN graphName
"""

DETECT_GLOBAL_ENTITY_HUB_COMMUNITIES = """
CALL gds.graph.project(
    $graph_name,
    'SourceEntityHub',
    {SIMILAR_TO: {orientation: 'UNDIRECTED', properties: 'score'}}
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
WITH gds.util.asNode(nodeId) AS node, values.communityIds AS communityIds
UNWIND communityIds AS community_id
RETURN community_id,
       node.uuid AS uuid,
       node.canonical_name AS canonical_name,
       node.description AS description
ORDER BY community_id, uuid
"""

REPLACE_GLOBAL_ENTITY_HUBS = """
CALL {
    MATCH (old_hub:GlobalEntityHub)
    DETACH DELETE old_hub
    RETURN count(*) AS deleted
}
UNWIND $hubs AS hub_data
CREATE (hub:GlobalEntityHub {
    uuid: hub_data.uuid,
    canonical_name: hub_data.canonical_name,
    aliases: hub_data.aliases,
    description: hub_data.description,
    embedding: hub_data.embedding
})
WITH hub, hub_data
UNWIND hub_data.member_uuids AS member_uuid
MATCH (member:SourceEntityHub {uuid: member_uuid})
CREATE (member)-[:IN_GLOBAL_HUB]->(hub)
RETURN count(DISTINCT hub) AS persisted
"""
