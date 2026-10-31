"""Cypher statements for source propositions, evidence, embeddings, and hubs."""

REPLACE_SOURCE_PROPOSITIONS = """
MATCH (source:Source {uuid: $source_uuid})
CALL (source) {
    OPTIONAL MATCH (source)-[:HAS_FLASHCARD]->(card:SourceFlashcard)
    OPTIONAL MATCH (card)-[:HAS_CARD_REVIEW]->(review:UserCardReview)
    OPTIONAL MATCH (review)-[:HAS_REVIEW_EVENT]->(event:ReviewEvent)
    WITH collect(DISTINCT card) AS cards,
         collect(DISTINCT review) AS reviews,
         collect(DISTINCT event) AS events
    FOREACH (event IN events | DETACH DELETE event)
    FOREACH (review IN reviews | DETACH DELETE review)
    FOREACH (card IN cards | DETACH DELETE card)
    RETURN count(*) AS learning_artifacts_deleted
}
CALL (source) {
    OPTIONAL MATCH (source)-[:HAS_PAGE]->(:SourcePage)
        -[:CONTAINS_BLOCK]->(:SourceBlock)-[:MEMBER_OF]->(:SourceProposition)
        -[:IN_SOURCE_HUB]->(hub:SourcePropositionHub)
    WITH collect(DISTINCT hub) AS hubs
    FOREACH (hub IN hubs | DETACH DELETE hub)
    RETURN count(*) AS old_hubs_deleted
}
CALL (source) {
    OPTIONAL MATCH (source)-[:HAS_PAGE]->(:SourcePage)
        -[:CONTAINS_BLOCK]->(:SourceBlock)-[:MEMBER_OF]->(old:SourceProposition)
    WITH collect(DISTINCT old) AS old_propositions
    FOREACH (proposition IN old_propositions | DETACH DELETE proposition)
    RETURN count(*) AS old_occurrences_deleted
}
CALL (source) {
    UNWIND $propositions AS row
    CREATE (proposition:SourceProposition {
        uuid: row.uuid,
        text: row.text
    })
    WITH source, proposition, row
    MATCH (source)-[:HAS_PAGE]->(:SourcePage)
        -[:CONTAINS_BLOCK]->(target_block:SourceBlock {
            uuid: row.target_block_uuid
        })
    CREATE (target_block)-[:MEMBER_OF]->(proposition)
    RETURN count(DISTINCT proposition) AS created
}
RETURN source.uuid AS uuid
"""

READ_SOURCE_PROPOSITIONS = """
MATCH (source:Source {uuid: $source_uuid})-[:FIRST_BLOCK]->(
    first_block:SourceBlock
)
MATCH path = (first_block)-[:NEXT_BLOCK*0..]->(target_block:SourceBlock)
MATCH (target_block)-[:MEMBER_OF]->(proposition:SourceProposition)
RETURN proposition.uuid AS uuid,
       source.uuid AS source_uuid,
       proposition.text AS text,
       [{uuid: target_block.uuid,
         block_type: target_block.block_type,
         content: target_block.content}] AS target_blocks
ORDER BY length(path), proposition.uuid
"""

READ_SOURCE_PROPOSITION_HUB_CANDIDATES = """
MATCH (source:Source {uuid: $source_uuid})-[:HAS_PAGE]->(:SourcePage)
      -[:CONTAINS_BLOCK]->(:SourceBlock)-[:MEMBER_OF]->(query:SourceProposition)
WHERE query.embedding IS NOT NULL
CALL db.index.vector.queryNodes(
    'source_proposition_embedding', $candidate_limit, query.embedding
)
YIELD node AS candidate, score
MATCH (source)-[:HAS_PAGE]->(:SourcePage)-[:CONTAINS_BLOCK]->(:SourceBlock)
      -[:MEMBER_OF]->(candidate)
WHERE candidate.uuid <> query.uuid
  AND score >= $minimum_similarity
WITH CASE WHEN query.uuid < candidate.uuid THEN query ELSE candidate END AS left,
     CASE WHEN query.uuid < candidate.uuid THEN candidate ELSE query END AS right,
     score
WITH left, right, max(score) AS score
RETURN left.uuid AS left_uuid,
       left.text AS left_text,
       right.uuid AS right_uuid,
       right.text AS right_text,
       score
ORDER BY left_uuid, right_uuid
"""

READ_SOURCE_PROPOSITION_HUB_MEMBERS = """
MATCH (source:Source {uuid: $source_uuid})-[:FIRST_BLOCK]->(
    first_block:SourceBlock
)
MATCH path = (first_block)-[:NEXT_BLOCK*0..]->(target_block:SourceBlock)
MATCH (target_block)-[:MEMBER_OF]->(proposition:SourceProposition)
RETURN proposition.uuid AS uuid,
       proposition.text AS text
ORDER BY length(path), proposition.uuid
"""

REPLACE_SOURCE_PROPOSITION_HUBS = """
MATCH (source:Source {uuid: $source_uuid})
CALL (source) {
    OPTIONAL MATCH (source)-[:HAS_FLASHCARD]->(card:SourceFlashcard)
    OPTIONAL MATCH (card)-[:HAS_CARD_REVIEW]->(review:UserCardReview)
    OPTIONAL MATCH (review)-[:HAS_REVIEW_EVENT]->(event:ReviewEvent)
    WITH collect(DISTINCT card) AS cards,
         collect(DISTINCT review) AS reviews,
         collect(DISTINCT event) AS events
    FOREACH (event IN events | DETACH DELETE event)
    FOREACH (review IN reviews | DETACH DELETE review)
    FOREACH (card IN cards | DETACH DELETE card)
    RETURN count(*) AS learning_artifacts_deleted
}
CALL (source) {
    OPTIONAL MATCH (source)-[:HAS_PAGE]->(:SourcePage)
        -[:CONTAINS_BLOCK]->(:SourceBlock)-[:MEMBER_OF]->(:SourceProposition)
        -[:IN_SOURCE_HUB]->(old_hub:SourcePropositionHub)
    WITH collect(DISTINCT old_hub) AS old_hubs
    FOREACH (hub IN old_hubs | DETACH DELETE hub)
    RETURN count(*) AS deleted
}
CALL (source) {
    UNWIND $hubs AS hub_data
    CREATE (hub:SourcePropositionHub {uuid: hub_data.uuid})
    WITH source, hub, hub_data
    UNWIND hub_data.member_uuids AS member_uuid
    MATCH (source)-[:HAS_PAGE]->(:SourcePage)-[:CONTAINS_BLOCK]->(:SourceBlock)
          -[:MEMBER_OF]->(member:SourceProposition {uuid: member_uuid})
    CREATE (member)-[:IN_SOURCE_HUB]->(hub)
    RETURN count(DISTINCT hub) AS persisted
}
RETURN source.uuid AS uuid
"""

SEARCH_SOURCE_PROPOSITION_HUBS = """
MATCH (source:Source {uuid: $source_uuid})-[:HAS_PAGE]->(:SourcePage)
      -[:CONTAINS_BLOCK]->(:SourceBlock)-[:MEMBER_OF]->(member:SourceProposition)
      -[:IN_SOURCE_HUB]->(hub:SourcePropositionHub)
WHERE member.embedding IS NOT NULL
WITH source, hub, member,
     vector.similarity.cosine(member.embedding, $query_embedding) AS score
ORDER BY hub.uuid, score DESC, member.uuid
WITH source, hub, collect({
    uuid: member.uuid,
    text: member.text,
    score: score
})[0] AS best
MATCH (source)-[:HAS_PAGE]->(:SourcePage)-[:CONTAINS_BLOCK]->(target_block:SourceBlock)
      -[:MEMBER_OF]->(member:SourceProposition {uuid: best.uuid})
MATCH (source)-[:FIRST_BLOCK]->(first:SourceBlock)
MATCH path = (first)-[:NEXT_BLOCK*0..]->(target_block)
WITH source, hub, member, best, target_block, path
ORDER BY length(path)
WITH source, hub, member, best, collect(target_block.uuid) AS target_block_uuids
RETURN hub.uuid AS uuid,
       source.uuid AS source_uuid,
       member.uuid AS member_uuid,
       target_block_uuids,
       best.text AS text,
       best.score AS score
ORDER BY score DESC, uuid
LIMIT $top_k
"""

UPDATE_SOURCE_PROPOSITION_EMBEDDINGS = """
UNWIND $rows AS row
MATCH (source:Source {uuid: $source_uuid})-[:HAS_PAGE]->(:SourcePage)
      -[:CONTAINS_BLOCK]->(:SourceBlock)-[:MEMBER_OF]->(
          proposition:SourceProposition {uuid: row.uuid}
      )
SET proposition.embedding = row.embedding
RETURN count(DISTINCT proposition) AS updated
"""
