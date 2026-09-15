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
