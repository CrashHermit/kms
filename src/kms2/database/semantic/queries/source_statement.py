"""Cypher statements for source statement."""

READ_SOURCE_STATEMENTS = """
MATCH (statement:SourceStatement {source_uuid: $source_uuid})
RETURN statement.uuid AS uuid, statement.source_uuid AS source_uuid,
       [(block:SourceBlock)-[:MEMBER_OF]->(statement) | block.uuid] AS member_block_uuids,
       statement.is_exercise AS is_exercise
ORDER BY uuid
"""

UPDATE_SOURCE_STATEMENT_DESCRIPTION = """
UNWIND $rows AS row
MATCH (statement:SourceStatement {uuid: row.uuid, source_uuid: $source_uuid})
SET statement.description = row.description, statement.embedding = row.embedding
RETURN count(statement) AS updated
"""

READ_SOURCE_STATEMENT_HUB_CANDIDATES = """
MATCH (query:SourceStatement {source_uuid: $source_uuid})
WHERE query.embedding IS NOT NULL
CALL db.index.vector.queryNodes('source_statement_embedding', $candidate_limit, query.embedding)
YIELD node AS candidate, score
WHERE candidate.source_uuid = $source_uuid AND candidate.uuid <> query.uuid AND score >= $minimum_similarity
WITH CASE WHEN query.uuid < candidate.uuid THEN query ELSE candidate END AS left,
     CASE WHEN query.uuid < candidate.uuid THEN candidate ELSE query END AS right, score
WITH left, right, max(score) AS score
RETURN left.uuid AS left_uuid, left.description AS left_description, right.uuid AS right_uuid,
       right.description AS right_description, score
ORDER BY left_uuid, right_uuid
"""

REPLACE_SOURCE_STATEMENT_ACCEPTED_EDGES = """
MATCH (left:SourceStatement {source_uuid: $source_uuid})-[old_similarity:SIMILAR_TO]-(right:SourceStatement {source_uuid: $source_uuid})
WHERE left.uuid < right.uuid DELETE old_similarity
WITH count(*) AS deleted
UNWIND $pairs AS pair
MATCH (left:SourceStatement {uuid: pair.left_uuid, source_uuid: $source_uuid})
MATCH (right:SourceStatement {uuid: pair.right_uuid, source_uuid: $source_uuid})
MERGE (left)-[similarity:SIMILAR_TO]->(right) SET similarity.score = pair.score
RETURN count(*) AS materialized
"""

DROP_SOURCE_STATEMENT_HUB_GRAPH = """CALL gds.graph.drop($graph_name, false) YIELD graphName RETURN graphName"""

DETECT_SOURCE_STATEMENT_COMMUNITIES = """
CALL gds.graph.project($graph_name, 'SourceStatement', {SIMILAR_TO: {orientation: 'UNDIRECTED', properties: 'score'}})
YIELD graphName
CALL gds.sllpa.stream($graph_name, {relationshipWeightProperty: 'score', maxIterations: $max_iterations, minAssociationStrength: $min_association_strength})
YIELD nodeId, values
WITH gds.util.asNode(nodeId) AS node, values.communityIds AS communityIds
UNWIND communityIds AS community_id
MATCH (node:SourceStatement {source_uuid: $source_uuid})
RETURN community_id, node.uuid AS uuid, node.description AS description
ORDER BY community_id, uuid
"""

REPLACE_SOURCE_STATEMENT_HUBS = """
CALL { MATCH (old_hub:SourceStatementHub {source_uuid: $source_uuid}) DETACH DELETE old_hub RETURN count(*) AS deleted }
UNWIND $hubs AS hub_data
CREATE (hub:SourceStatementHub {uuid: hub_data.uuid, source_uuid: $source_uuid, canonical_name: hub_data.canonical_name, aliases: hub_data.aliases, description: hub_data.description, embedding: hub_data.embedding})
WITH hub, hub_data
UNWIND hub_data.member_uuids AS member_uuid
MATCH (member:SourceStatement {uuid: member_uuid, source_uuid: $source_uuid})
CREATE (member)-[:IN_SOURCE_HUB]->(hub)
RETURN count(DISTINCT hub) AS persisted
"""
