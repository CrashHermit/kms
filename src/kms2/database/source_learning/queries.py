"""Cypher statements for source-learning evidence and artifacts."""

_SOURCE_SCOPE = """
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
"""

READ_SOURCE_ENTITY_LEARNING_INPUTS = (
    _SOURCE_SCOPE
    + """
MATCH (triplet)-[member_role:HAS_SUBJECT|HAS_OBJECT]->(
    member:SourceEntity {source_uuid: $source_uuid}
)
MATCH (member)-[:IN_SOURCE_HUB]->(hub:SourceEntityHub {
    source_uuid: $source_uuid
})
WITH DISTINCT hub, fact, triplet, subject, predicate, object,
     type(member_role) AS member_role, length(source_path) AS source_position
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
       hub.canonical_name AS hub_name,
       hub.description AS hub_description,
       evidence
ORDER BY hub_uuid
"""
)

READ_SOURCE_EVENT_LEARNING_INPUTS = (
    _SOURCE_SCOPE
    + """
MATCH (triplet)-[member_role:HAS_SUBJECT|HAS_OBJECT]->(
    member:SourceEvent {source_uuid: $source_uuid}
)
MATCH (member)-[:IN_SOURCE_HUB]->(hub:SourceEventHub {
    source_uuid: $source_uuid
})
WITH DISTINCT hub, fact, triplet, subject, predicate, object,
     type(member_role) AS member_role, length(source_path) AS source_position
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
)

READ_SOURCE_PREDICATE_LEARNING_INPUTS = (
    _SOURCE_SCOPE
    + """
MATCH (predicate)-[:IN_SOURCE_HUB]->(hub:SourcePredicateHub {
    source_uuid: $source_uuid
})
WITH DISTINCT hub, fact, triplet, subject, predicate, object,
     length(source_path) AS source_position
ORDER BY source_position, hub.uuid, fact.uuid, triplet.uuid
WITH hub, collect({
    source_fact_uuid: fact.uuid,
    source_fact_text: fact.text,
    triplet_uuid: triplet.uuid,
    subject: subject.name,
    predicate: predicate.predicate,
    object: object.name,
    member_role: 'predicate'
}) AS evidence
RETURN hub.uuid AS hub_uuid,
       hub.predicate AS hub_name,
       hub.description AS hub_description,
       evidence
ORDER BY hub_uuid
"""
)

READ_SOURCE_TRIPLET_LEARNING_INPUTS = (
    _SOURCE_SCOPE
    + """
MATCH (triplet)-[:IN_SOURCE_HUB]->(hub:SourceTripletHub {
    source_uuid: $source_uuid
})
WITH DISTINCT hub, fact, triplet, subject, predicate, object,
     length(source_path) AS source_position
ORDER BY source_position, hub.uuid, fact.uuid, triplet.uuid
WITH hub, collect({
    source_fact_uuid: fact.uuid,
    source_fact_text: fact.text,
    triplet_uuid: triplet.uuid,
    subject: subject.name,
    predicate: predicate.predicate,
    object: object.name,
    member_role: 'triplet'
}) AS evidence
RETURN hub.uuid AS hub_uuid,
       hub.canonical_name AS hub_name,
       hub.description AS hub_description,
       evidence
ORDER BY hub_uuid
"""
)

_CLEAR_SOURCE_LEARNING = """
MATCH (node)
WHERE node.source_uuid = $source_uuid
  AND (
      node:SourceFlashcard
      OR node:SourceEntityLearningFact
      OR node:SourceEventLearningFact
      OR node:SourcePredicateLearningFact
      OR node:SourceTripletLearningFact
  )
DETACH DELETE node
RETURN count(node) AS deleted
"""

REPLACE_SOURCE_LEARNING_FACTS = {
    'entity': """
UNWIND $rows AS row
CREATE (learning_fact:SourceEntityLearningFact {
    uuid: row.uuid, source_uuid: $source_uuid, text: row.text
})
WITH learning_fact, row
MATCH (hub:SourceEntityHub {uuid: row.hub_uuid, source_uuid: $source_uuid})
CREATE (learning_fact)-[:ABOUT_HUB]->(hub)
WITH learning_fact, row
UNWIND row.source_fact_uuids AS source_fact_uuid
MATCH (source_fact:SourceFact {uuid: source_fact_uuid})
CREATE (learning_fact)-[:SUPPORTED_BY]->(source_fact)
WITH learning_fact, row
UNWIND row.triplet_uuids AS triplet_uuid
MATCH (triplet:SourceTriplet {uuid: triplet_uuid})
CREATE (learning_fact)-[:SUPPORTED_BY_TRIPLET]->(triplet)
RETURN count(DISTINCT learning_fact) AS persisted
""",
    'event': """
UNWIND $rows AS row
CREATE (learning_fact:SourceEventLearningFact {
    uuid: row.uuid, source_uuid: $source_uuid, text: row.text
})
WITH learning_fact, row
MATCH (hub:SourceEventHub {uuid: row.hub_uuid, source_uuid: $source_uuid})
CREATE (learning_fact)-[:ABOUT_HUB]->(hub)
WITH learning_fact, row
UNWIND row.source_fact_uuids AS source_fact_uuid
MATCH (source_fact:SourceFact {uuid: source_fact_uuid})
CREATE (learning_fact)-[:SUPPORTED_BY]->(source_fact)
WITH learning_fact, row
UNWIND row.triplet_uuids AS triplet_uuid
MATCH (triplet:SourceTriplet {uuid: triplet_uuid})
CREATE (learning_fact)-[:SUPPORTED_BY_TRIPLET]->(triplet)
RETURN count(DISTINCT learning_fact) AS persisted
""",
    'predicate': """
UNWIND $rows AS row
CREATE (learning_fact:SourcePredicateLearningFact {
    uuid: row.uuid, source_uuid: $source_uuid, text: row.text
})
WITH learning_fact, row
MATCH (hub:SourcePredicateHub {uuid: row.hub_uuid, source_uuid: $source_uuid})
CREATE (learning_fact)-[:ABOUT_HUB]->(hub)
WITH learning_fact, row
UNWIND row.source_fact_uuids AS source_fact_uuid
MATCH (source_fact:SourceFact {uuid: source_fact_uuid})
CREATE (learning_fact)-[:SUPPORTED_BY]->(source_fact)
WITH learning_fact, row
UNWIND row.triplet_uuids AS triplet_uuid
MATCH (triplet:SourceTriplet {uuid: triplet_uuid})
CREATE (learning_fact)-[:SUPPORTED_BY_TRIPLET]->(triplet)
RETURN count(DISTINCT learning_fact) AS persisted
""",
    'triplet': """
UNWIND $rows AS row
CREATE (learning_fact:SourceTripletLearningFact {
    uuid: row.uuid, source_uuid: $source_uuid, text: row.text
})
WITH learning_fact, row
MATCH (hub:SourceTripletHub {uuid: row.hub_uuid, source_uuid: $source_uuid})
CREATE (learning_fact)-[:ABOUT_HUB]->(hub)
WITH learning_fact, row
UNWIND row.source_fact_uuids AS source_fact_uuid
MATCH (source_fact:SourceFact {uuid: source_fact_uuid})
CREATE (learning_fact)-[:SUPPORTED_BY]->(source_fact)
WITH learning_fact, row
UNWIND row.triplet_uuids AS triplet_uuid
MATCH (triplet:SourceTriplet {uuid: triplet_uuid})
CREATE (learning_fact)-[:SUPPORTED_BY_TRIPLET]->(triplet)
RETURN count(DISTINCT learning_fact) AS persisted
""",
}

PERSIST_SOURCE_FLASHCARDS = """
UNWIND $rows AS row
CREATE (card:SourceFlashcard {
    uuid: row.uuid, source_uuid: $source_uuid,
    question: row.question, answer: row.answer
})
WITH card, row
MATCH (learning_fact {uuid: row.learning_fact_uuid, source_uuid: $source_uuid})
WHERE learning_fact:SourceEntityLearningFact
   OR learning_fact:SourceEventLearningFact
   OR learning_fact:SourcePredicateLearningFact
   OR learning_fact:SourceTripletLearningFact
CREATE (card)-[:DERIVED_FROM]->(learning_fact)
RETURN count(card) AS persisted
"""

_READ_LEARNING_FACTS = {
    'entity': """
MATCH (learning_fact:SourceEntityLearningFact {source_uuid: $source_uuid})
MATCH (learning_fact)-[:ABOUT_HUB]->(hub:SourceEntityHub)
MATCH (learning_fact)-[:SUPPORTED_BY]->(source_fact:SourceFact)
MATCH (source_fact)-[:HAS_TRIPLET]->(triplet:SourceTriplet)
MATCH (triplet)-[:HAS_SUBJECT]->(subject)
MATCH (triplet)-[:HAS_PREDICATE]->(predicate:SourcePredicate)
MATCH (triplet)-[:HAS_OBJECT]->(object)
WITH learning_fact, hub, collect(DISTINCT {
    source_fact_uuid: source_fact.uuid,
    source_fact_text: source_fact.text,
    triplet_uuid: triplet.uuid,
    subject: subject.name,
    predicate: predicate.predicate,
    object: object.name,
    member_role: CASE
        WHEN EXISTS { MATCH (triplet)-[:HAS_SUBJECT]->(:SourceEntity)-[:IN_SOURCE_HUB]->(hub) }
        THEN 'subject' ELSE 'object' END
}) AS evidence
RETURN learning_fact.uuid AS learning_fact_uuid,
       learning_fact.text AS learning_fact_text,
       hub.uuid AS hub_uuid, hub.canonical_name AS hub_name,
       hub.description AS hub_description, evidence
ORDER BY learning_fact_uuid
""",
    'event': """
MATCH (learning_fact:SourceEventLearningFact {source_uuid: $source_uuid})
MATCH (learning_fact)-[:ABOUT_HUB]->(hub:SourceEventHub)
MATCH (learning_fact)-[:SUPPORTED_BY]->(source_fact:SourceFact)
MATCH (source_fact)-[:HAS_TRIPLET]->(triplet:SourceTriplet)
MATCH (triplet)-[:HAS_SUBJECT]->(subject)
MATCH (triplet)-[:HAS_PREDICATE]->(predicate:SourcePredicate)
MATCH (triplet)-[:HAS_OBJECT]->(object)
WITH learning_fact, hub, collect(DISTINCT {
    source_fact_uuid: source_fact.uuid,
    source_fact_text: source_fact.text,
    triplet_uuid: triplet.uuid,
    subject: subject.name,
    predicate: predicate.predicate,
    object: object.name,
    member_role: CASE
        WHEN EXISTS { MATCH (triplet)-[:HAS_SUBJECT]->(:SourceEvent)-[:IN_SOURCE_HUB]->(hub) }
        THEN 'subject' ELSE 'object' END
}) AS evidence
RETURN learning_fact.uuid AS learning_fact_uuid,
       learning_fact.text AS learning_fact_text,
       hub.uuid AS hub_uuid, hub.name AS hub_name,
       hub.description AS hub_description, evidence
ORDER BY learning_fact_uuid
""",
    'predicate': """
MATCH (learning_fact:SourcePredicateLearningFact {source_uuid: $source_uuid})
MATCH (learning_fact)-[:ABOUT_HUB]->(hub:SourcePredicateHub)
MATCH (learning_fact)-[:SUPPORTED_BY]->(source_fact:SourceFact)
MATCH (source_fact)-[:HAS_TRIPLET]->(triplet:SourceTriplet)
MATCH (triplet)-[:HAS_SUBJECT]->(subject)
MATCH (triplet)-[:HAS_PREDICATE]->(predicate:SourcePredicate)
MATCH (triplet)-[:HAS_OBJECT]->(object)
WITH learning_fact, hub, collect(DISTINCT {
    source_fact_uuid: source_fact.uuid,
    source_fact_text: source_fact.text,
    triplet_uuid: triplet.uuid,
    subject: subject.name,
    predicate: predicate.predicate,
    object: object.name,
    member_role: 'predicate'
}) AS evidence
RETURN learning_fact.uuid AS learning_fact_uuid,
       learning_fact.text AS learning_fact_text,
       hub.uuid AS hub_uuid, hub.predicate AS hub_name,
       hub.description AS hub_description, evidence
ORDER BY learning_fact_uuid
""",
    'triplet': """
MATCH (learning_fact:SourceTripletLearningFact {source_uuid: $source_uuid})
MATCH (learning_fact)-[:ABOUT_HUB]->(hub:SourceTripletHub)
MATCH (learning_fact)-[:SUPPORTED_BY]->(source_fact:SourceFact)
MATCH (source_fact)-[:HAS_TRIPLET]->(triplet:SourceTriplet)
MATCH (triplet)-[:HAS_SUBJECT]->(subject)
MATCH (triplet)-[:HAS_PREDICATE]->(predicate:SourcePredicate)
MATCH (triplet)-[:HAS_OBJECT]->(object)
WITH learning_fact, hub, collect(DISTINCT {
    source_fact_uuid: source_fact.uuid,
    source_fact_text: source_fact.text,
    triplet_uuid: triplet.uuid,
    subject: subject.name,
    predicate: predicate.predicate,
    object: object.name,
    member_role: 'triplet'
}) AS evidence
RETURN learning_fact.uuid AS learning_fact_uuid,
       learning_fact.text AS learning_fact_text,
       hub.uuid AS hub_uuid, hub.canonical_name AS hub_name,
       hub.description AS hub_description, evidence
ORDER BY learning_fact_uuid
""",
}

__all__ = [
    'PERSIST_SOURCE_FLASHCARDS',
    'READ_SOURCE_ENTITY_LEARNING_INPUTS',
    'READ_SOURCE_EVENT_LEARNING_INPUTS',
    'READ_SOURCE_PREDICATE_LEARNING_INPUTS',
    'READ_SOURCE_TRIPLET_LEARNING_INPUTS',
    'REPLACE_SOURCE_LEARNING_FACTS',
    '_CLEAR_SOURCE_LEARNING',
    '_READ_LEARNING_FACTS',
]
