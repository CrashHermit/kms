"""Cypher statements for global triplets and global triplet hubs."""

REPLACE_GLOBAL_TRIPLETS = """
CALL {
    MATCH (old_triplet:GlobalTriplet)
    DETACH DELETE old_triplet
    RETURN count(*) AS deleted
}
CALL {
    MATCH (source_triplet_hub:SourceTripletHub)-[:HAS_SUBJECT_HUB]->(
        source_subject_hub
    )
    MATCH (source_triplet_hub)-[:HAS_PREDICATE_HUB]->(
        source_predicate_hub:SourcePredicateHub
    )
    MATCH (source_triplet_hub)-[:HAS_OBJECT_HUB]->(source_object_hub)
    WHERE (source_subject_hub:SourceEntityHub OR source_subject_hub:SourceEventHub)
      AND (source_object_hub:SourceEntityHub OR source_object_hub:SourceEventHub)
    MATCH (source_subject_hub)-[:IN_GLOBAL_HUB]->(subject_hub)
    WHERE (source_subject_hub:SourceEntityHub AND subject_hub:GlobalEntityHub)
       OR (source_subject_hub:SourceEventHub AND subject_hub:GlobalEventHub)
    MATCH (source_predicate_hub)-[:IN_GLOBAL_HUB]->(
        predicate_hub:GlobalPredicateHub
    )
    MATCH (source_object_hub)-[:IN_GLOBAL_HUB]->(object_hub)
    WHERE (source_object_hub:SourceEntityHub AND object_hub:GlobalEntityHub)
       OR (source_object_hub:SourceEventHub AND object_hub:GlobalEventHub)
    CREATE (global_triplet:GlobalTriplet {uuid: randomUUID()})
    CREATE (source_triplet_hub)-[:IN_GLOBAL_TRIPLET]->(global_triplet)
    CREATE (global_triplet)-[:HAS_SUBJECT_HUB]->(subject_hub)
    CREATE (global_triplet)-[:HAS_PREDICATE_HUB]->(predicate_hub)
    CREATE (global_triplet)-[:HAS_OBJECT_HUB]->(object_hub)
    RETURN count(DISTINCT global_triplet) AS persisted
}
RETURN persisted
"""

READ_GLOBAL_TRIPLET_HUB_GROUPS = """
MATCH (global_triplet:GlobalTriplet)-[:HAS_SUBJECT_HUB]->(subject_hub)
WHERE subject_hub:GlobalEntityHub OR subject_hub:GlobalEventHub
MATCH (global_triplet)-[:HAS_PREDICATE_HUB]->(
    predicate_hub:GlobalPredicateHub
)
MATCH (global_triplet)-[:HAS_OBJECT_HUB]->(object_hub)
WHERE object_hub:GlobalEntityHub OR object_hub:GlobalEventHub
MATCH (source_triplet_hub:SourceTripletHub)-[:IN_GLOBAL_TRIPLET]->(
    global_triplet
)
WITH subject_hub, predicate_hub, object_hub,
     collect(DISTINCT global_triplet.uuid) AS global_triplet_uuids,
     collect(DISTINCT {
         global_triplet_uuid: global_triplet.uuid,
         source_triplet_hub_uuid: source_triplet_hub.uuid,
         canonical_name: source_triplet_hub.canonical_name,
         description: source_triplet_hub.description
     }) AS evidence
RETURN
    subject_hub.uuid AS subject_hub_uuid,
    CASE WHEN subject_hub:GlobalEntityHub
         THEN subject_hub.canonical_name
         ELSE subject_hub.name END AS subject_hub_name,
    subject_hub.description AS subject_hub_description,
    predicate_hub.uuid AS predicate_hub_uuid,
    predicate_hub.predicate AS predicate_hub_name,
    predicate_hub.description AS predicate_hub_description,
    object_hub.uuid AS object_hub_uuid,
    CASE WHEN object_hub:GlobalEntityHub
         THEN object_hub.canonical_name
         ELSE object_hub.name END AS object_hub_name,
    object_hub.description AS object_hub_description,
    global_triplet_uuids,
    evidence
ORDER BY subject_hub_uuid, predicate_hub_uuid, object_hub_uuid
"""

REPLACE_GLOBAL_TRIPLET_HUBS = """
CALL {
    MATCH (old_hub:GlobalTripletHub)
    DETACH DELETE old_hub
}
WITH 1 AS _
UNWIND $hubs AS hub_data
CREATE (hub:GlobalTripletHub {
    uuid: hub_data.uuid,
    canonical_name: hub_data.canonical_name,
    description: hub_data.description,
    embedding: hub_data.embedding,
    subject_hub_uuid: hub_data.subject_hub_uuid,
    predicate_hub_uuid: hub_data.predicate_hub_uuid,
    object_hub_uuid: hub_data.object_hub_uuid
})
WITH hub, hub_data
MATCH (subject_hub {uuid: hub_data.subject_hub_uuid})
WHERE subject_hub:GlobalEntityHub OR subject_hub:GlobalEventHub
MATCH (predicate_hub:GlobalPredicateHub {
    uuid: hub_data.predicate_hub_uuid
})
MATCH (object_hub {uuid: hub_data.object_hub_uuid})
WHERE object_hub:GlobalEntityHub OR object_hub:GlobalEventHub
CREATE (hub)-[:HAS_SUBJECT_HUB]->(subject_hub)
CREATE (hub)-[:HAS_PREDICATE_HUB]->(predicate_hub)
CREATE (hub)-[:HAS_OBJECT_HUB]->(object_hub)
WITH hub, hub_data
UNWIND hub_data.global_triplet_uuids AS global_triplet_uuid
MATCH (global_triplet:GlobalTriplet {uuid: global_triplet_uuid})
CREATE (global_triplet)-[:IN_GLOBAL_HUB]->(hub)
RETURN count(DISTINCT hub) AS persisted
"""
