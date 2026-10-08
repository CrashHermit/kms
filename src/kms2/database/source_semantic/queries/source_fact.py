"""Cypher statements for durable node-first source facts."""

REPLACE_SOURCE_FACTS = """
MATCH (source:Source {uuid: $source_uuid})
CALL (source) {
    OPTIONAL MATCH (source)-[:HAS_FLASHCARD]->(card:SourceFlashcard)
    OPTIONAL MATCH (card)-[:HAS_CARD_REVIEW]->(review:UserCardReview)
    OPTIONAL MATCH (review)-[:HAS_REVIEW_EVENT]->(event:ReviewEvent)
    WITH source,
         collect(DISTINCT card) AS cards,
         collect(DISTINCT review) AS reviews,
         collect(DISTINCT event) AS events
    FOREACH (event IN events | DETACH DELETE event)
    FOREACH (review IN reviews | DETACH DELETE review)
    FOREACH (card IN cards | DETACH DELETE card)
    RETURN size(cards) AS learning_artifacts_deleted
}
WITH source
CALL (source) {
    MATCH (source)-[:FIRST_BLOCK]->(first:SourceBlock)
    MATCH (first)-[:NEXT_BLOCK*0..]->(source_block:SourceBlock)
    MATCH (source_block)<-[:HAS_SOURCE_BLOCK]-(target:SourceFactTarget)
          <-[:HAS_TARGET]-(old_fact:SourceFact)
    OPTIONAL MATCH (old_fact)-[
        :HAS_TARGET|HAS_CONTEXT_BEFORE|HAS_CONTEXT_AFTER
    ]->(old_pointer)
    OPTIONAL MATCH (old_fact)-[:HAS_TRIPLET]->(old_triplet:SourceTriplet)
    OPTIONAL MATCH (old_triplet)-[:HAS_SUBJECT|HAS_OBJECT]->(old_endpoint)
    OPTIONAL MATCH (old_triplet)-[:HAS_PREDICATE]->(
        old_predicate:SourcePredicate
    )
    WITH collect(DISTINCT old_fact) AS old_facts,
         collect(DISTINCT old_pointer) AS old_pointers,
         collect(DISTINCT old_triplet) AS old_triplets,
         collect(DISTINCT old_endpoint) AS old_endpoints,
         collect(DISTINCT old_predicate) AS old_predicates
    FOREACH (node IN old_triplets | DETACH DELETE node)
    FOREACH (node IN old_facts | DETACH DELETE node)
    FOREACH (node IN old_pointers | DETACH DELETE node)
    FOREACH (node IN old_endpoints | DETACH DELETE node)
    FOREACH (node IN old_predicates | DETACH DELETE node)
    RETURN count(*) AS old_facts_deleted
}
WITH source
UNWIND $facts AS row
CREATE (fact:SourceFact {
    uuid: row.uuid,
    text: row.text
})
CREATE (target:SourceFactTarget {uuid: row.target_uuid})
CREATE (before:SourceFactContext {uuid: row.context_before_uuid})
CREATE (after:SourceFactContext {uuid: row.context_after_uuid})
CREATE (fact)-[:HAS_TARGET]->(target)
CREATE (fact)-[:HAS_CONTEXT_BEFORE]->(before)
CREATE (fact)-[:HAS_CONTEXT_AFTER]->(after)
WITH source,
     fact,
     target,
     before,
     after,
     row
CALL (target, row) {
    UNWIND row.target_block_uuids AS block_uuid
    MATCH (block:SourceBlock {uuid: block_uuid})
    CREATE (target)-[:HAS_SOURCE_BLOCK]->(block)
    RETURN count(*) AS target_blocks_persisted
}
CALL (before, row) {
    UNWIND row.context_before_block_uuids AS block_uuid
    MATCH (block:SourceBlock {uuid: block_uuid})
    CREATE (before)-[:HAS_SOURCE_BLOCK]->(block)
    RETURN count(*) AS context_before_blocks_persisted
}
CALL (after, row) {
    UNWIND row.context_after_block_uuids AS block_uuid
    MATCH (block:SourceBlock {uuid: block_uuid})
    CREATE (after)-[:HAS_SOURCE_BLOCK]->(block)
    RETURN count(*) AS context_after_blocks_persisted
}
RETURN source.uuid AS uuid
"""

READ_SOURCE_FACTS = """
MATCH (source:Source {uuid: $source_uuid})-[:FIRST_BLOCK]->(first:SourceBlock)
MATCH source_path = (first)-[:NEXT_BLOCK*0..]->(source_block:SourceBlock)
MATCH (source_block)<-[:HAS_SOURCE_BLOCK]-(target:SourceFactTarget)
      <-[:HAS_TARGET]-(fact:SourceFact)
WITH DISTINCT source,
     fact,
     target
CALL (source, target) {
    MATCH (source)-[:FIRST_BLOCK]->(first:SourceBlock)
    MATCH path = (first)-[:NEXT_BLOCK*0..]->(block:SourceBlock)
    WHERE (target)-[:HAS_SOURCE_BLOCK]->(block)
    WITH block,
         path
    ORDER BY length(path)
    RETURN collect({
        uuid: block.uuid,
        block_type: block.block_type,
        content: block.content
    }) AS target_blocks,
           min(length(path)) AS target_position
}
RETURN fact.uuid AS uuid,
       fact.text AS text,
       target.uuid AS target_uuid,
       target_blocks
ORDER BY target_position, uuid
"""

READ_SOURCE_FACT_CONTEXTS = """
MATCH (source:Source {uuid: $source_uuid})-[:FIRST_BLOCK]->(first:SourceBlock)
MATCH source_path = (first)-[:NEXT_BLOCK*0..]->(source_block:SourceBlock)
MATCH (source_block)<-[:HAS_SOURCE_BLOCK]-(target:SourceFactTarget)
      <-[:HAS_TARGET]-(fact:SourceFact)
WITH DISTINCT source,
     fact
MATCH (fact)-[context_edge:HAS_CONTEXT_BEFORE|HAS_CONTEXT_AFTER]->(
    context:SourceFactContext
)
WITH source,
     fact,
     type(context_edge) AS context_kind,
     context
CALL (source, context) {
    MATCH (source)-[:FIRST_BLOCK]->(first:SourceBlock)
    MATCH path = (first)-[:NEXT_BLOCK*0..]->(block:SourceBlock)
    WHERE (context)-[:HAS_SOURCE_BLOCK]->(block)
    WITH block,
         path
    ORDER BY length(path)
    RETURN collect({
        uuid: block.uuid,
        block_type: block.block_type,
        content: block.content
    }) AS blocks
}
RETURN fact.uuid AS fact_uuid,
       context_kind,
       context.uuid AS context_uuid,
       blocks
ORDER BY fact_uuid, context_kind
"""
