"""Cypher statements for replacing raw semantic assertions and searching vertices."""

REPLACE_SOURCE_ASSERTIONS = """
MATCH (source:Source {uuid: $source_uuid})
OPTIONAL MATCH (old_triplet:Triplet {source_uuid: $source_uuid})
OPTIONAL MATCH (old_triplet)-[:HAS_SUBJECT|HAS_OBJECT]->(old_endpoint)
OPTIONAL MATCH (old_triplet)-[:HAS_PREDICATE]->(old_source_predicate:SourcePredicate)
WITH source,
     collect(DISTINCT old_triplet) AS old_triplets,
     collect(DISTINCT old_endpoint) AS old_endpoints,
     collect(DISTINCT old_source_predicate) AS old_source_predicates
FOREACH (node IN old_triplets | DETACH DELETE node)
FOREACH (node IN old_endpoints | DETACH DELETE node)
FOREACH (node IN old_source_predicates | DETACH DELETE node)
WITH source
CALL (source) {
    UNWIND $triplets AS row
    CREATE (triplet:Triplet {
        uuid: row.uuid,
        source_uuid: row.source_uuid,
        source_block_uuid: row.source_block_uuid,
        subject_uuid: row.subject_uuid,
        object_uuid: row.object_uuid,
        predicate_uuid: row.predicate_uuid
    })
    RETURN count(*) AS _
}
WITH source
CALL (source) {
    UNWIND $source_entities AS row
    CREATE (source_entity:SourceEntity {
        uuid: row.uuid,
        source_uuid: row.source_uuid,
        source_block_uuid: row.source_block_uuid,
        name: row.name
    })
    RETURN count(*) AS _
}
WITH source
CALL (source) {
    UNWIND $source_events AS row
    CREATE (source_event:SourceEvent {
        uuid: row.uuid,
        source_uuid: row.source_uuid,
        source_block_uuid: row.source_block_uuid,
        name: row.name
    })
    RETURN count(*) AS _
}
WITH source
CALL (source) {
    UNWIND $source_predicates AS row
    CREATE (source_predicate:SourcePredicate {
        uuid: row.uuid,
        source_uuid: row.source_uuid,
        source_block_uuid: row.source_block_uuid,
        predicate: row.predicate
    })
    RETURN count(*) AS _
}
WITH source
CALL (source) {
    UNWIND $triplets AS row
    MATCH (triplet:Triplet {uuid: row.uuid})
    MATCH (block:SourceBlock {uuid: row.source_block_uuid})
    CREATE (block)-[:HAS_TRIPLET]->(triplet)
    RETURN count(*) AS _
}
WITH source
CALL (source) {
    UNWIND $triplets AS row
    MATCH (triplet:Triplet {uuid: row.uuid})
    MATCH (subject {uuid: row.subject_uuid})
    MATCH (object {uuid: row.object_uuid})
    MATCH (source_predicate:SourcePredicate {uuid: row.predicate_uuid})
    CREATE (triplet)-[:HAS_SUBJECT]->(subject)
    CREATE (triplet)-[:HAS_OBJECT]->(object)
    CREATE (triplet)-[:HAS_PREDICATE]->(source_predicate)
    RETURN count(*) AS _
}
RETURN source.uuid AS uuid
"""

CLEAR_SOURCE_ASSERTIONS = """
MATCH (source:Source {uuid: $source_uuid})
OPTIONAL MATCH (old_triplet:Triplet {source_uuid: $source_uuid})
OPTIONAL MATCH (old_triplet)-[:HAS_SUBJECT|HAS_OBJECT]->(old_endpoint)
OPTIONAL MATCH (old_triplet)-[:HAS_PREDICATE]->(old_source_predicate:SourcePredicate)
WITH collect(DISTINCT old_triplet) AS old_triplets,
     collect(DISTINCT old_endpoint) AS old_endpoints,
     collect(DISTINCT old_source_predicate) AS old_source_predicates
FOREACH (node IN old_triplets | DETACH DELETE node)
FOREACH (node IN old_endpoints | DETACH DELETE node)
FOREACH (node IN old_source_predicates | DETACH DELETE node)
RETURN count(*) AS cleared
"""

READ_SOURCE_ENTITIES = """
MATCH (source_entity:SourceEntity {source_uuid: $source_uuid})
RETURN source_entity.uuid AS uuid,
       source_entity.source_uuid AS source_uuid,
       source_entity.source_block_uuid AS source_block_uuid,
       source_entity.name AS name
ORDER BY source_entity.source_block_uuid, source_entity.uuid
"""

READ_SOURCE_EVENTS = """
MATCH (source_event:SourceEvent {source_uuid: $source_uuid})
RETURN source_event.uuid AS uuid,
       source_event.source_uuid AS source_uuid,
       source_event.source_block_uuid AS source_block_uuid,
       source_event.name AS name
ORDER BY source_event.source_block_uuid, source_event.uuid
"""

READ_SOURCE_PREDICATES = """
MATCH (source_predicate:SourcePredicate {source_uuid: $source_uuid})
RETURN source_predicate.uuid AS uuid,
       source_predicate.source_uuid AS source_uuid,
       source_predicate.source_block_uuid AS source_block_uuid,
       source_predicate.predicate AS predicate
ORDER BY source_predicate.source_block_uuid, source_predicate.uuid
"""

READ_SOURCE_STATEMENTS = """
MATCH (statement:Statement {source_uuid: $source_uuid})
RETURN statement.uuid AS uuid, statement.source_uuid AS source_uuid,
       [(block:SourceBlock)-[:MEMBER_OF]->(statement) | block.uuid] AS member_block_uuids,
       statement.is_exercise AS is_exercise
ORDER BY uuid
"""

READ_SOURCE_PROCEDURES = """
MATCH (procedure:Procedure {source_uuid: $source_uuid})
RETURN procedure.uuid AS uuid, procedure.source_uuid AS source_uuid,
       [(block:SourceBlock)-[:MEMBER_OF]->(procedure) | block.uuid] AS member_block_uuids
ORDER BY uuid
"""

UPDATE_SOURCE_STATEMENT_DESCRIPTION = """
UNWIND $rows AS row
MATCH (statement:Statement {uuid: row.uuid, source_uuid: $source_uuid})
SET statement.description = row.description, statement.embedding = row.embedding
RETURN count(statement) AS updated
"""

UPDATE_SOURCE_PROCEDURE_DESCRIPTION = """
UNWIND $rows AS row
MATCH (procedure:Procedure {uuid: row.uuid, source_uuid: $source_uuid})
SET procedure.description = row.description, procedure.embedding = row.embedding
RETURN count(procedure) AS updated
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

READ_SOURCE_PREDICATE_HUB_CANDIDATES = """
MATCH (query:SourcePredicate {source_uuid: $source_uuid})
WHERE query.embedding IS NOT NULL
CALL db.index.vector.queryNodes('source_predicate_embedding', $candidate_limit, query.embedding)
YIELD node AS candidate, score
WHERE candidate.source_uuid = $source_uuid AND candidate.uuid <> query.uuid AND score >= $minimum_similarity
WITH CASE WHEN query.uuid < candidate.uuid THEN query ELSE candidate END AS left,
     CASE WHEN query.uuid < candidate.uuid THEN candidate ELSE query END AS right, score
WITH left, right, max(score) AS score
MATCH (left_triplet:Triplet)-[:HAS_PREDICATE]->(left)
MATCH (left_triplet)-[:HAS_SUBJECT]->(left_subject)
MATCH (left_triplet)-[:HAS_OBJECT]->(left_object)
MATCH (right_triplet:Triplet)-[:HAS_PREDICATE]->(right)
MATCH (right_triplet)-[:HAS_SUBJECT]->(right_subject)
MATCH (right_triplet)-[:HAS_OBJECT]->(right_object)
RETURN left.uuid AS left_uuid, left.predicate AS left_predicate, left.description AS left_description,
       left_subject.name AS left_subject, left_object.name AS left_object,
       right.uuid AS right_uuid, right.predicate AS right_predicate, right.description AS right_description,
       right_subject.name AS right_subject, right_object.name AS right_object, score
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

REPLACE_SOURCE_EVENT_ACCEPTED_EDGES = (
    REPLACE_SOURCE_ENTITY_ACCEPTED_EDGES.replace('SourceEntity', 'SourceEvent')
)
REPLACE_SOURCE_PREDICATE_ACCEPTED_EDGES = (
    REPLACE_SOURCE_ENTITY_ACCEPTED_EDGES.replace(
        'SourceEntity', 'SourcePredicate'
    )
)

DROP_SOURCE_ENTITY_HUB_GRAPH = (
    'CALL gds.graph.drop($graph_name, false) YIELD graphName RETURN graphName'
)
DROP_SOURCE_EVENT_HUB_GRAPH = DROP_SOURCE_ENTITY_HUB_GRAPH
DROP_SOURCE_PREDICATE_HUB_GRAPH = DROP_SOURCE_ENTITY_HUB_GRAPH

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
DETECT_SOURCE_EVENT_COMMUNITIES = DETECT_SOURCE_ENTITY_COMMUNITIES.replace(
    'SourceEntity', 'SourceEvent'
).replace('node.name AS name', 'node.name AS name')
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
REPLACE_SOURCE_EVENT_HUBS = REPLACE_SOURCE_ENTITY_HUBS.replace(
    'SourceEntity', 'SourceEvent'
).replace('canonical_name: hub_data.canonical_name', 'name: hub_data.name')
REPLACE_SOURCE_PREDICATE_HUBS = REPLACE_SOURCE_ENTITY_HUBS.replace(
    'SourceEntity', 'SourcePredicate'
).replace(
    'canonical_name: hub_data.canonical_name', 'predicate: hub_data.predicate'
)

READ_SOURCE_STATEMENT_HUB_CANDIDATES = """
MATCH (query:Statement {source_uuid: $source_uuid})
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
READ_SOURCE_PROCEDURE_HUB_CANDIDATES = (
    READ_SOURCE_STATEMENT_HUB_CANDIDATES.replace(
        'Statement', 'Procedure'
    ).replace('statement', 'procedure')
)

REPLACE_SOURCE_STATEMENT_ACCEPTED_EDGES = (
    REPLACE_SOURCE_ENTITY_ACCEPTED_EDGES.replace('SourceEntity', 'Statement')
)
REPLACE_SOURCE_PROCEDURE_ACCEPTED_EDGES = (
    REPLACE_SOURCE_ENTITY_ACCEPTED_EDGES.replace('SourceEntity', 'Procedure')
)
DROP_SOURCE_STATEMENT_HUB_GRAPH = DROP_SOURCE_ENTITY_HUB_GRAPH
DROP_SOURCE_PROCEDURE_HUB_GRAPH = DROP_SOURCE_ENTITY_HUB_GRAPH
DETECT_SOURCE_STATEMENT_COMMUNITIES = DETECT_SOURCE_ENTITY_COMMUNITIES.replace(
    'SourceEntity', 'Statement'
).replace('node.name AS name, ', '')
DETECT_SOURCE_PROCEDURE_COMMUNITIES = (
    DETECT_SOURCE_STATEMENT_COMMUNITIES.replace('Statement', 'Procedure')
)
REPLACE_SOURCE_STATEMENT_HUBS = REPLACE_SOURCE_ENTITY_HUBS.replace(
    'SourceEntityHub', 'SourceStatementHub'
).replace('SourceEntity', 'Statement')
REPLACE_SOURCE_PROCEDURE_HUBS = REPLACE_SOURCE_ENTITY_HUBS.replace(
    'SourceEntityHub', 'SourceProcedureHub'
).replace('SourceEntity', 'Procedure')
