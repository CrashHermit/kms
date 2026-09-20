"""Cypher statements for source triplet."""

REPLACE_SOURCE_FACTS_AND_TRIPLETS = """
MATCH (source:Source {uuid: $source_uuid})
OPTIONAL MATCH (old_triplet:SourceTriplet {source_uuid: $source_uuid})
OPTIONAL MATCH (old_fact:SourceFact {source_uuid: $source_uuid})
OPTIONAL MATCH (old_triplet)-[:HAS_SUBJECT|HAS_OBJECT]->(old_endpoint)
OPTIONAL MATCH (old_triplet)-[:HAS_PREDICATE]->(old_source_predicate:SourcePredicate)
WITH source,
     collect(DISTINCT old_triplet) AS old_triplets,
     collect(DISTINCT old_fact) AS old_facts,
     collect(DISTINCT old_endpoint) AS old_endpoints,
     collect(DISTINCT old_source_predicate) AS old_source_predicates
FOREACH (node IN old_triplets | DETACH DELETE node)
FOREACH (node IN old_facts | DETACH DELETE node)
FOREACH (node IN old_endpoints | DETACH DELETE node)
FOREACH (node IN old_source_predicates | DETACH DELETE node)
WITH source
CALL (source) {
    UNWIND $source_facts AS row
    CREATE (fact:SourceFact {
        uuid: row.uuid,
        source_uuid: row.source_uuid,
        source_block_uuid: row.source_block_uuid,
        text: row.text
    })
    WITH fact, row
    MATCH (block:SourceBlock {uuid: row.source_block_uuid})
    CREATE (block)-[:HAS_FACT]->(fact)
    RETURN count(*) AS _
}
WITH source
CALL (source) {
    UNWIND $triplets AS row
    CREATE (triplet:SourceTriplet {
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
    UNWIND $fact_triplet_pairs AS row
    MATCH (fact:SourceFact {uuid: row.source_fact_uuid})
    MATCH (triplet:SourceTriplet {uuid: row.triplet_uuid})
    CREATE (fact)-[:HAS_TRIPLET]->(triplet)
    RETURN count(*) AS _
}
WITH source
CALL (source) {
    UNWIND $triplets AS row
    MATCH (triplet:SourceTriplet {uuid: row.uuid})
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

CLEAR_SOURCE_FACTS_AND_TRIPLETS = """
MATCH (source:Source {uuid: $source_uuid})
OPTIONAL MATCH (old_triplet:SourceTriplet {source_uuid: $source_uuid})
OPTIONAL MATCH (old_fact:SourceFact {source_uuid: $source_uuid})
OPTIONAL MATCH (old_triplet)-[:HAS_SUBJECT|HAS_OBJECT]->(old_endpoint)
OPTIONAL MATCH (old_triplet)-[:HAS_PREDICATE]->(old_source_predicate:SourcePredicate)
WITH collect(DISTINCT old_triplet) AS old_triplets,
     collect(DISTINCT old_fact) AS old_facts,
     collect(DISTINCT old_endpoint) AS old_endpoints,
     collect(DISTINCT old_source_predicate) AS old_source_predicates
FOREACH (node IN old_triplets | DETACH DELETE node)
FOREACH (node IN old_facts | DETACH DELETE node)
FOREACH (node IN old_endpoints | DETACH DELETE node)
FOREACH (node IN old_source_predicates | DETACH DELETE node)
RETURN count(*) AS cleared
"""

READ_SOURCE_TRIPLET_HUB_GROUPS = """
MATCH (fact:SourceFact {source_uuid: $source_uuid})-[:HAS_TRIPLET]->(
    triplet:SourceTriplet {source_uuid: $source_uuid}
)
MATCH (triplet)-[:HAS_SUBJECT]->(subject)
WHERE (subject:SourceEntity OR subject:SourceEvent)
  AND subject.source_uuid = $source_uuid
MATCH (subject)-[:IN_SOURCE_HUB]->(subject_hub)
WHERE (subject_hub:SourceEntityHub OR subject_hub:SourceEventHub)
  AND subject_hub.source_uuid = $source_uuid
MATCH (triplet)-[:HAS_PREDICATE]->(
    predicate:SourcePredicate {source_uuid: $source_uuid}
)
MATCH (predicate)-[:IN_SOURCE_HUB]->(
    predicate_hub:SourcePredicateHub {source_uuid: $source_uuid}
)
MATCH (triplet)-[:HAS_OBJECT]->(object)
WHERE (object:SourceEntity OR object:SourceEvent)
  AND object.source_uuid = $source_uuid
MATCH (object)-[:IN_SOURCE_HUB]->(object_hub)
WHERE (object_hub:SourceEntityHub OR object_hub:SourceEventHub)
  AND object_hub.source_uuid = $source_uuid
WITH subject_hub, predicate_hub, object_hub,
     collect(DISTINCT triplet.uuid) AS triplet_uuids,
     collect(DISTINCT {
         triplet_uuid: triplet.uuid,
         fact_text: fact.text,
         subject: subject.name,
         predicate: predicate.predicate,
         object: object.name
     }) AS evidence
RETURN
    subject_hub.uuid AS subject_hub_uuid,
    CASE WHEN subject_hub:SourceEntityHub
         THEN subject_hub.canonical_name
         ELSE subject_hub.name END AS subject_hub_name,
    subject_hub.description AS subject_hub_description,
    predicate_hub.uuid AS predicate_hub_uuid,
    predicate_hub.predicate AS predicate_hub_name,
    predicate_hub.description AS predicate_hub_description,
    object_hub.uuid AS object_hub_uuid,
    CASE WHEN object_hub:SourceEntityHub
         THEN object_hub.canonical_name
         ELSE object_hub.name END AS object_hub_name,
    object_hub.description AS object_hub_description,
    triplet_uuids,
    evidence
ORDER BY subject_hub_uuid, predicate_hub_uuid, object_hub_uuid
"""

REPLACE_SOURCE_TRIPLET_HUBS = """
CALL {
    MATCH (old_hub:SourceTripletHub {source_uuid: $source_uuid})
    DETACH DELETE old_hub
}
WITH 1 AS _
UNWIND $hubs AS hub_data
CREATE (hub:SourceTripletHub {
    uuid: hub_data.uuid,
    source_uuid: $source_uuid,
    canonical_name: hub_data.canonical_name,
    description: hub_data.description,
    embedding: hub_data.embedding,
    subject_hub_uuid: hub_data.subject_hub_uuid,
    predicate_hub_uuid: hub_data.predicate_hub_uuid,
    object_hub_uuid: hub_data.object_hub_uuid
})
WITH hub, hub_data
MATCH (subject_hub {uuid: hub_data.subject_hub_uuid,
                    source_uuid: $source_uuid})
WHERE subject_hub:SourceEntityHub OR subject_hub:SourceEventHub
MATCH (predicate_hub:SourcePredicateHub {
    uuid: hub_data.predicate_hub_uuid,
    source_uuid: $source_uuid
})
MATCH (object_hub {uuid: hub_data.object_hub_uuid,
                   source_uuid: $source_uuid})
WHERE object_hub:SourceEntityHub OR object_hub:SourceEventHub
CREATE (hub)-[:HAS_SUBJECT_HUB]->(subject_hub)
CREATE (hub)-[:HAS_PREDICATE_HUB]->(predicate_hub)
CREATE (hub)-[:HAS_OBJECT_HUB]->(object_hub)
WITH hub, hub_data
UNWIND hub_data.triplet_uuids AS triplet_uuid
MATCH (triplet:SourceTriplet {uuid: triplet_uuid, source_uuid: $source_uuid})
CREATE (triplet)-[:IN_SOURCE_HUB]->(hub)
RETURN count(DISTINCT hub) AS persisted
"""
