"""Source-scoped predicate learning queries."""

READ_SOURCE_PREDICATE_LEARNING_REQUESTS = """
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
MATCH (predicate)-[:IN_SOURCE_HUB]->(hub:SourcePredicateHub {source_uuid: $source_uuid})
WITH hub, fact, min(length(source_path)) AS source_position
RETURN hub.uuid AS hub_uuid,
       hub.predicate AS hub_name,
       fact.uuid AS source_fact_uuid,
       fact.text AS source_fact_text
ORDER BY source_position, hub_uuid, source_fact_uuid
"""

CREATE_SOURCE_PREDICATE_LEARNING_FACTS = """
UNWIND $rows AS row
MATCH (source:Source {uuid: $source_uuid})
MATCH (hub:SourcePredicateHub {uuid: row.hub_uuid, source_uuid: $source_uuid})
MATCH (source_fact:SourceFact {uuid: row.source_fact_uuid})
CREATE (source)-[:HAS_LEARNING_FACT]->(learning_fact:SourcePredicateLearningFact {
    uuid: row.uuid,
    text: row.text
})
CREATE (learning_fact)-[:ABOUT_HUB]->(hub)
CREATE (learning_fact)-[:SUPPORTED_BY]->(source_fact)
RETURN count(learning_fact) AS persisted
"""

READ_SOURCE_PREDICATE_FLASHCARD_REQUESTS = """
MATCH (source:Source {uuid: $source_uuid})-[:HAS_LEARNING_FACT]->
      (learning_fact:SourcePredicateLearningFact)
RETURN learning_fact.uuid AS learning_fact_uuid,
       learning_fact.text AS learning_fact_text
ORDER BY learning_fact_uuid
"""
