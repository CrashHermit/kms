"""Cypher statements for replacing raw semantic assertions."""

REPLACE_SOURCE_ASSERTIONS = """
MATCH (source:Source {uuid: $source_uuid})
OPTIONAL MATCH (old_triplet:Triplet {source_uuid: $source_uuid})
OPTIONAL MATCH (old_triplet)-[:HAS_SUBJECT|HAS_OBJECT]->(old_endpoint)
OPTIONAL MATCH (old_triplet)-[:HAS_PREDICATE]->(old_predicate:Predicate)
WITH source,
     collect(DISTINCT old_triplet) AS old_triplets,
     collect(DISTINCT old_endpoint) AS old_endpoints,
     collect(DISTINCT old_predicate) AS old_predicates
FOREACH (node IN old_triplets | DETACH DELETE node)
FOREACH (node IN old_endpoints | DETACH DELETE node)
FOREACH (node IN old_predicates | DETACH DELETE node)
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
    UNWIND $entities AS row
    CREATE (entity:Entity {
        uuid: row.uuid,
        source_uuid: row.source_uuid,
        source_block_uuid: row.source_block_uuid,
        name: row.name
    })
    RETURN count(*) AS _
}
WITH source
CALL (source) {
    UNWIND $events AS row
    CREATE (event:Event {
        uuid: row.uuid,
        source_uuid: row.source_uuid,
        source_block_uuid: row.source_block_uuid,
        name: row.name
    })
    RETURN count(*) AS _
}
WITH source
CALL (source) {
    UNWIND $predicates AS row
    CREATE (predicate:Predicate {
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
    MATCH (predicate:Predicate {uuid: row.predicate_uuid})
    CREATE (triplet)-[:HAS_SUBJECT]->(subject)
    CREATE (triplet)-[:HAS_OBJECT]->(object)
    CREATE (triplet)-[:HAS_PREDICATE]->(predicate)
    RETURN count(*) AS _
}
RETURN source.uuid AS uuid
"""

CLEAR_SOURCE_ASSERTIONS = """
MATCH (source:Source {uuid: $source_uuid})
OPTIONAL MATCH (old_triplet:Triplet {source_uuid: $source_uuid})
OPTIONAL MATCH (old_triplet)-[:HAS_SUBJECT|HAS_OBJECT]->(old_endpoint)
OPTIONAL MATCH (old_triplet)-[:HAS_PREDICATE]->(old_predicate:Predicate)
WITH collect(DISTINCT old_triplet) AS old_triplets,
     collect(DISTINCT old_endpoint) AS old_endpoints,
     collect(DISTINCT old_predicate) AS old_predicates
FOREACH (node IN old_triplets | DETACH DELETE node)
FOREACH (node IN old_endpoints | DETACH DELETE node)
FOREACH (node IN old_predicates | DETACH DELETE node)
RETURN count(*) AS cleared
"""


READ_SOURCE_ENTITIES = """
MATCH (entity:Entity {source_uuid: $source_uuid})
RETURN entity.uuid AS uuid,
       entity.source_uuid AS source_uuid,
       entity.source_block_uuid AS source_block_uuid,
       entity.name AS name
ORDER BY entity.source_block_uuid, entity.uuid
"""

READ_SOURCE_EVENTS = """
MATCH (event:Event {source_uuid: $source_uuid})
RETURN event.uuid AS uuid,
       event.source_uuid AS source_uuid,
       event.source_block_uuid AS source_block_uuid,
       event.name AS name
ORDER BY event.source_block_uuid, event.uuid
"""

READ_SOURCE_PREDICATES = """
MATCH (predicate:Predicate {source_uuid: $source_uuid})
RETURN predicate.uuid AS uuid,
       predicate.source_uuid AS source_uuid,
       predicate.source_block_uuid AS source_block_uuid,
       predicate.predicate AS predicate
ORDER BY predicate.source_block_uuid, predicate.uuid
"""

UPDATE_ENTITY_ENRICHMENT = """
UNWIND $rows AS row
MATCH (entity:Entity {uuid: row.uuid, source_uuid: $source_uuid})
SET entity.description = row.description,
    entity.embedding = row.embedding
RETURN count(entity) AS updated
"""

UPDATE_EVENT_ENRICHMENT = """
UNWIND $rows AS row
MATCH (event:Event {uuid: row.uuid, source_uuid: $source_uuid})
SET event.description = row.description,
    event.embedding = row.embedding
RETURN count(event) AS updated
"""

UPDATE_PREDICATE_ENRICHMENT = """
UNWIND $rows AS row
MATCH (predicate:Predicate {uuid: row.uuid, source_uuid: $source_uuid})
SET predicate.description = row.description,
    predicate.embedding = row.embedding
RETURN count(predicate) AS updated
"""
