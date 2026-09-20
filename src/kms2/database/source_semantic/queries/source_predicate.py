"""Cypher statements for source predicate."""

READ_SOURCE_PREDICATES = """
MATCH (source_predicate:SourcePredicate {source_uuid: $source_uuid})
RETURN source_predicate.uuid AS uuid,
       source_predicate.source_uuid AS source_uuid,
       source_predicate.source_block_uuid AS source_block_uuid,
       source_predicate.predicate AS predicate
ORDER BY source_predicate.source_block_uuid, source_predicate.uuid
"""

UPDATE_SOURCE_PREDICATE_DESCRIPTION = """
UNWIND $rows AS row
MATCH (source_predicate:SourcePredicate {
    uuid: row.uuid,
    source_uuid: $source_uuid
})
SET source_predicate.description = row.description,
    source_predicate.embedding = row.embedding
RETURN count(source_predicate) AS updated
"""

FIND_SIMILAR_SOURCE_PREDICATES = """
MATCH (query:SourcePredicate {uuid: $query_uuid})
CALL db.index.vector.queryNodes(
    'source_predicate_embedding',
    $candidate_limit,
    query.embedding
)
YIELD node, score
WHERE node.uuid <> query.uuid
RETURN node.uuid AS uuid,
       node.source_uuid AS source_uuid,
       node.source_block_uuid AS source_block_uuid,
       node.predicate AS predicate,
       node.description AS description,
       score
ORDER BY score DESC, uuid ASC
LIMIT $top_k
"""

READ_SOURCE_PREDICATE_HUB_CANDIDATES = """
MATCH (query:SourcePredicate {source_uuid: $source_uuid})
WHERE query.embedding IS NOT NULL
CALL db.index.vector.queryNodes('source_predicate_embedding', $candidate_limit, query.embedding)
YIELD node AS candidate, score
WHERE candidate.source_uuid = $source_uuid AND candidate.uuid <> query.uuid AND score >= $minimum_similarity
WITH CASE WHEN query.uuid < candidate.uuid THEN query ELSE candidate END AS left,
     CASE WHEN query.uuid < candidate.uuid THEN candidate ELSE query END AS right, score
WITH left, right, max(score) AS score
MATCH (left_triplet:SourceTriplet)-[:HAS_PREDICATE]->(left)
MATCH (left_triplet)-[:HAS_SUBJECT]->(left_subject)
MATCH (left_triplet)-[:HAS_OBJECT]->(left_object)
MATCH (right_triplet:SourceTriplet)-[:HAS_PREDICATE]->(right)
MATCH (right_triplet)-[:HAS_SUBJECT]->(right_subject)
MATCH (right_triplet)-[:HAS_OBJECT]->(right_object)
RETURN left.uuid AS left_uuid, left.predicate AS left_predicate, left.description AS left_description,
       left_subject.name AS left_subject, left_object.name AS left_object,
       right.uuid AS right_uuid, right.predicate AS right_predicate, right.description AS right_description,
       right_subject.name AS right_subject, right_object.name AS right_object, score
ORDER BY left_uuid, right_uuid
"""

REPLACE_SOURCE_PREDICATE_ACCEPTED_EDGES = """
MATCH (left:SourcePredicate {source_uuid: $source_uuid})-[old_similarity:SIMILAR_TO]-(right:SourcePredicate {source_uuid: $source_uuid})
WHERE left.uuid < right.uuid DELETE old_similarity
WITH count(*) AS deleted
UNWIND $pairs AS pair
MATCH (left:SourcePredicate {uuid: pair.left_uuid, source_uuid: $source_uuid})
MATCH (right:SourcePredicate {uuid: pair.right_uuid, source_uuid: $source_uuid})
MERGE (left)-[similarity:SIMILAR_TO]->(right) SET similarity.score = pair.score
RETURN count(*) AS materialized
"""

DROP_SOURCE_PREDICATE_HUB_GRAPH = """CALL gds.graph.drop($graph_name, false) YIELD graphName RETURN graphName"""

DETECT_SOURCE_PREDICATE_COMMUNITIES = """
CALL gds.graph.project($graph_name, 'SourcePredicate', {SIMILAR_TO: {orientation: 'UNDIRECTED', properties: 'score'}})
YIELD graphName
CALL gds.sllpa.stream($graph_name, {relationshipWeightProperty: 'score', maxIterations: $max_iterations, minAssociationStrength: $min_association_strength})
YIELD nodeId, values
WITH gds.util.asNode(nodeId) AS node, values.communityIds AS communityIds
UNWIND communityIds AS community_id
MATCH (node:SourcePredicate {source_uuid: $source_uuid})
RETURN community_id, node.uuid AS uuid, node.predicate AS predicate, node.description AS description
ORDER BY community_id, uuid
"""

REPLACE_SOURCE_PREDICATE_HUBS = """
CALL { MATCH (old_hub:SourcePredicateHub {source_uuid: $source_uuid}) DETACH DELETE old_hub RETURN count(*) AS deleted }
UNWIND $hubs AS hub_data
CREATE (hub:SourcePredicateHub {uuid: hub_data.uuid, source_uuid: $source_uuid, predicate: hub_data.predicate, aliases: hub_data.aliases, description: hub_data.description, embedding: hub_data.embedding})
WITH hub, hub_data
UNWIND hub_data.member_uuids AS member_uuid
MATCH (member:SourcePredicate {uuid: member_uuid, source_uuid: $source_uuid})
CREATE (member)-[:IN_SOURCE_HUB]->(hub)
RETURN count(DISTINCT hub) AS persisted
"""
