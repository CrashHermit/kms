"""Source-scoped event learning queries.

Readers retain each selected source-fact/triplet tuple once, including both
endpoint roles where applicable. Writers persist one learning-fact node per
supplied row and link only its selected source facts and triplets.
"""

READ_SOURCE_EVENT_LEARNING_INPUTS = """
MATCH (source:Source {uuid: $source_uuid})-[:FIRST_BLOCK]->(first:SourceBlock)
MATCH source_path = (first)-[:NEXT_BLOCK*0..]->(source_block:SourceBlock)
MATCH (source_block)<-[:HAS_SOURCE_BLOCK]-(:SourceFactTarget)
    <-[:HAS_TARGET]-(fact:SourceFact)-[:HAS_TRIPLET]->(triplet:SourceTriplet)
MATCH (triplet)-[:HAS_SUBJECT]->(subject)
WHERE (subject:SourceEntity OR subject:SourceEvent)
  AND subject.source_uuid = $source_uuid
MATCH (triplet)-[:HAS_PREDICATE]->(predicate:SourcePredicate {source_uuid: $source_uuid})
MATCH (triplet)-[:HAS_OBJECT]->(object)
WHERE (object:SourceEntity OR object:SourceEvent)
  AND object.source_uuid = $source_uuid

MATCH (triplet)-[member_role:HAS_SUBJECT|HAS_OBJECT]->(
    member:SourceEvent {source_uuid: $source_uuid}
)
MATCH (member)-[:IN_SOURCE_HUB]->(hub:SourceEventHub {
    source_uuid: $source_uuid
})
WITH hub, fact, triplet, subject, predicate, object,
     CASE type(member_role)
         WHEN 'HAS_SUBJECT' THEN 'subject'
         ELSE 'object'
     END AS member_role,
     min(length(source_path)) AS source_position
ORDER BY source_position, hub.uuid, fact.uuid, triplet.uuid, member_role
WITH hub, collect({
    source_fact_uuid: fact.uuid,
    source_fact_text: fact.text,
    triplet_uuid: triplet.uuid,
    subject: subject.name,
    predicate: predicate.predicate,
    object: object.name,
    member_role: member_role
}) AS evidence
RETURN hub.uuid AS hub_uuid,
       hub.name AS hub_name,
       hub.description AS hub_description,
       evidence
ORDER BY hub_uuid
"""

CREATE_SOURCE_EVENT_LEARNING_FACTS = """
UNWIND $rows AS row
MATCH (source:Source {uuid: $source_uuid})
CREATE (source)-[:HAS_LEARNING_FACT]->(learning_fact:SourceEventLearningFact {
    uuid: row.uuid, text: row.text
})
WITH learning_fact, row
MATCH (hub:SourceEventHub {uuid: row.hub_uuid, source_uuid: $source_uuid})
CREATE (learning_fact)-[:ABOUT_HUB]->(hub)
WITH learning_fact, row
CALL (learning_fact, row) {
    UNWIND row.source_fact_uuids AS source_fact_uuid
    MATCH (source_fact:SourceFact {uuid: source_fact_uuid})
    CREATE (learning_fact)-[:SUPPORTED_BY]->(source_fact)
    RETURN count(source_fact) AS source_facts_persisted
}
CALL (learning_fact, row) {
    UNWIND row.triplet_uuids AS triplet_uuid
    MATCH (triplet:SourceTriplet {uuid: triplet_uuid})
    CREATE (learning_fact)-[:SUPPORTED_BY_TRIPLET]->(triplet)
    RETURN count(triplet) AS triplets_persisted
}
RETURN count(DISTINCT learning_fact) AS persisted
"""

READ_SOURCE_EVENT_FLASHCARD_INPUTS = """
MATCH (source:Source {uuid: $source_uuid})-[:HAS_LEARNING_FACT]->
      (learning_fact:SourceEventLearningFact)
MATCH (learning_fact)-[:ABOUT_HUB]->(hub:SourceEventHub)
MATCH (learning_fact)-[:SUPPORTED_BY]->(source_fact:SourceFact)
MATCH (learning_fact)-[:SUPPORTED_BY_TRIPLET]->(triplet:SourceTriplet)
MATCH (source_fact)-[:HAS_TRIPLET]->(triplet)
MATCH (triplet)-[member_role:HAS_SUBJECT|HAS_OBJECT]->(
    member:SourceEvent
)
MATCH (member)-[:IN_SOURCE_HUB]->(hub)
MATCH (triplet)-[:HAS_SUBJECT]->(subject)
MATCH (triplet)-[:HAS_PREDICATE]->(predicate:SourcePredicate)
MATCH (triplet)-[:HAS_OBJECT]->(object)
WITH DISTINCT learning_fact, hub, source_fact, triplet,
     subject, predicate, object,
     CASE type(member_role)
         WHEN 'HAS_SUBJECT' THEN 'subject'
         ELSE 'object'
     END AS member_role
ORDER BY source_fact.uuid, triplet.uuid, member_role
WITH learning_fact, hub, collect({
    source_fact_uuid: source_fact.uuid,
    source_fact_text: source_fact.text,
    triplet_uuid: triplet.uuid,
    subject: subject.name,
    predicate: predicate.predicate,
    object: object.name,
    member_role: member_role
}) AS evidence
RETURN learning_fact.uuid AS learning_fact_uuid,
       learning_fact.text AS learning_fact_text,
       hub.uuid AS hub_uuid, hub.name AS hub_name,
       hub.description AS hub_description, evidence
ORDER BY learning_fact_uuid
"""
