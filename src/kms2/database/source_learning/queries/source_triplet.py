"""Cypher statements for loading source-learning triplet context."""

READ_SOURCE_ATOMIC_FLASHCARD_REQUESTS = """
MATCH (source:Source {uuid: $source_uuid})-[:FIRST_BLOCK]->(first:SourceBlock)
MATCH source_path=(first)-[:NEXT_BLOCK*0..]->(source_block:SourceBlock)
MATCH (source_block)<-[:HAS_SOURCE_BLOCK]-(:SourceFactTarget)
    <-[:HAS_TARGET]-(fact:SourceFact)-[:HAS_TRIPLET]->(triplet:SourceTriplet)
MATCH (triplet)-[:HAS_SUBJECT]->(subject)
WHERE (subject:SourceEntity OR subject:SourceEvent)
  AND subject.source_uuid = $source_uuid
MATCH (triplet)-[:HAS_PREDICATE]->(
    predicate:SourcePredicate {source_uuid: $source_uuid}
)
MATCH (triplet)-[:HAS_OBJECT]->(object)
WHERE (object:SourceEntity OR object:SourceEvent)
  AND object.source_uuid = $source_uuid
MATCH (triplet)-[:IN_SOURCE_HUB]->(
    triplet_hub:SourceTripletHub {source_uuid: $source_uuid}
)
MATCH (triplet_hub)-[:HAS_SUBJECT_HUB]->(subject_hub)
WHERE (subject_hub:SourceEntityHub OR subject_hub:SourceEventHub)
  AND subject_hub.source_uuid = $source_uuid
MATCH (triplet_hub)-[:HAS_PREDICATE_HUB]->(
    predicate_hub:SourcePredicateHub {source_uuid: $source_uuid}
)
MATCH (triplet_hub)-[:HAS_OBJECT_HUB]->(object_hub)
WHERE (object_hub:SourceEntityHub OR object_hub:SourceEventHub)
  AND object_hub.source_uuid = $source_uuid
WITH min(length(source_path)) AS source_position,
     triplet_hub,
     subject_hub,
     predicate_hub,
     object_hub,
     triplet,
     fact,
     subject,
     predicate,
     object
RETURN source_position,
       triplet_hub.uuid AS hub_uuid,
       triplet.uuid AS triplet_uuid,
       fact.uuid AS source_fact_uuid,
       CASE WHEN triplet_hub.canonical_name IS NOT NULL
            THEN triplet_hub.canonical_name
            ELSE triplet_hub.name END AS triplet_hub_name,
       triplet_hub.description AS triplet_hub_description,
       CASE WHEN subject_hub.canonical_name IS NOT NULL
            THEN subject_hub.canonical_name
            ELSE subject_hub.name END AS subject_hub_name,
       subject_hub.description AS subject_hub_description,
       predicate_hub.predicate AS predicate_hub_name,
       predicate_hub.description AS predicate_hub_description,
       CASE WHEN object_hub.canonical_name IS NOT NULL
            THEN object_hub.canonical_name
            ELSE object_hub.name END AS object_hub_name,
       object_hub.description AS object_hub_description,
       subject.name AS subject,
       predicate.predicate AS predicate,
       object.name AS object,
       fact.text AS source_fact_text
ORDER BY source_position, hub_uuid, triplet_uuid, source_fact_uuid
"""
