"""Owned, bounded knowledge reads and proposition search."""

READ_KNOWLEDGE_REVISION = """
MATCH (user:User {uuid: $user_uuid})
RETURN coalesce(user.knowledge_revision, 0) AS revision
"""

SEARCH_OWNED_PROPOSITIONS = """
CALL db.index.vector.queryNodes(
    'proposition_embedding',
    $candidate_limit,
    $query_embedding
)
YIELD node AS proposition, score AS similarity
MATCH (:User {uuid: $user_uuid})-[:HAS_PROPOSITION]->(proposition)
RETURN {
    uuid: proposition.uuid,
    content: proposition.content,
    embedding: proposition.embedding,
    created_at: proposition.created_at
} AS proposition,
similarity
ORDER BY similarity DESC, proposition.uuid ASC
"""

SEARCH_OWNED_ARGUMENTS = """
CALL db.index.vector.queryNodes(
    'argument_embedding',
    $candidate_limit,
    $query_embedding
)
YIELD node AS argument, score AS similarity
MATCH (user:User {uuid: $user_uuid})-[:HAS_ARGUMENT]->(argument)
MATCH (user)-[:HAS_PROPOSITION]->(conclusion:Proposition)
      <-[:PROVES]-(argument)
CALL (user, argument) {
    OPTIONAL MATCH (premise:Proposition)-[:PREMISE_OF]->(argument)
    WHERE (user)-[:HAS_PROPOSITION]->(premise)
    RETURN [
        item IN collect(DISTINCT CASE WHEN premise IS NULL THEN NULL ELSE {
            uuid: premise.uuid,
            content: premise.content
        } END)
        WHERE item IS NOT NULL
    ] AS premises
}
CALL (user, argument) {
    OPTIONAL MATCH (argument)-[:SUPPORTED_BY]->(block:SourceBlock)
    WHERE EXISTS {
        MATCH (user)-[:OWNS_SOURCE]->(:Source)-[:HAS_PAGE]->
              (:SourcePage)-[:CONTAINS_BLOCK]->(block)
    }
    RETURN collect(DISTINCT block.uuid) AS supporting_block_uuids
}
RETURN argument AS argument,
       [premise IN premises | premise.uuid] AS premise_uuids,
       conclusion.uuid AS conclusion_uuid,
       [premise IN premises | premise.content] AS premise_contents,
       conclusion.content AS conclusion_content,
       supporting_block_uuids,
       similarity
ORDER BY similarity DESC, argument.uuid ASC
"""

READ_OWNED_PROPOSITION = """
MATCH (user:User {uuid: $user_uuid})-[:HAS_PROPOSITION]->(
    proposition:Proposition {uuid: $proposition_uuid}
)
CALL (user, proposition) {
    OPTIONAL MATCH (user)-[:HAS_TEACHING_SESSION]->(:TeachingSession)
          -[:HAS_TURN]->(turn:TeachingTurn)
          -[:HAS_KNOWLEDGE_ASSERTION]->(assertion:KnowledgeAssertion)
          -[:ASSERTS]->(proposition)
    WITH DISTINCT turn, assertion
    CALL (user, assertion) {
        OPTIONAL MATCH (assertion)-[:SUPPORTED_BY]->(block:SourceBlock)
        WHERE EXISTS {
            MATCH (user)-[:OWNS_SOURCE]->(:Source)-[:HAS_PAGE]->
                  (:SourcePage)-[:CONTAINS_BLOCK]->(block)
        }
        RETURN collect(DISTINCT block.uuid) AS supporting_block_uuids
    }
    CALL (user, assertion) {
        OPTIONAL MATCH (assertion)-[:DERIVED_BY]->(argument:Argument)
        WHERE (user)-[:HAS_ARGUMENT]->(argument)
        RETURN collect(DISTINCT argument.uuid) AS argument_uuids
    }
    RETURN [
        item IN collect(CASE WHEN assertion IS NULL THEN NULL ELSE {
            assertion: assertion,
            proposition_uuid: proposition.uuid,
            supporting_block_uuids: supporting_block_uuids,
            argument_uuids: argument_uuids,
            teaching_turn_uuid: turn.uuid
        } END)
        WHERE item IS NOT NULL
    ] AS assertions
}
CALL (user, proposition) {
    OPTIONAL MATCH (user)-[:HAS_ARGUMENT]->
          (argument:Argument)-[:PROVES]->(proposition)
    RETURN collect(DISTINCT argument.uuid) AS argument_uuids
}
RETURN proposition AS proposition, assertions, argument_uuids
"""

READ_OWNED_ARGUMENT = """
MATCH (user:User {uuid: $user_uuid})-[:HAS_ARGUMENT]->(
    argument:Argument {uuid: $argument_uuid}
)
MATCH (user)-[:HAS_PROPOSITION]->(conclusion:Proposition)
      <-[:PROVES]-(argument)
CALL (user, argument) {
    OPTIONAL MATCH (premise:Proposition)-[:PREMISE_OF]->(argument)
    WHERE (user)-[:HAS_PROPOSITION]->(premise)
    RETURN collect(DISTINCT premise.uuid) AS premise_uuids
}
CALL (user, argument) {
    OPTIONAL MATCH (argument)-[:SUPPORTED_BY]->(block:SourceBlock)
    WHERE EXISTS {
        MATCH (user)-[:OWNS_SOURCE]->(:Source)-[:HAS_PAGE]->
              (:SourcePage)-[:CONTAINS_BLOCK]->(block)
    }
    RETURN collect(DISTINCT block.uuid) AS supporting_block_uuids
}
RETURN argument AS argument,
       premise_uuids,
       conclusion.uuid AS conclusion_uuid,
       supporting_block_uuids
"""

FIND_EXACT_OWNED_PROPOSITION = """
MATCH (:User {uuid: $user_uuid})-[:HAS_PROPOSITION]->(proposition:Proposition {
    content: $content
})
RETURN proposition AS proposition
ORDER BY proposition.created_at ASC, proposition.uuid ASC
LIMIT 1
"""

FIND_OWNED_ARGUMENT_CANDIDATES = """
MATCH (user:User {uuid: $user_uuid})-[:HAS_ARGUMENT]->(
    argument:Argument
)
MATCH (user)-[:HAS_PROPOSITION]->(conclusion:Proposition)
      <-[:PROVES]-(argument)
WHERE conclusion.uuid = $conclusion_uuid
  AND argument.explanation = $explanation
CALL (user, argument) {
    OPTIONAL MATCH (premise:Proposition)-[:PREMISE_OF]->(argument)
    WHERE (user)-[:HAS_PROPOSITION]->(premise)
    RETURN collect(DISTINCT premise.uuid) AS premise_uuids
}
CALL (user, argument) {
    OPTIONAL MATCH (argument)-[:SUPPORTED_BY]->(block:SourceBlock)
    WHERE EXISTS {
        MATCH (user)-[:OWNS_SOURCE]->(:Source)-[:HAS_PAGE]->
              (:SourcePage)-[:CONTAINS_BLOCK]->(block)
    }
    RETURN collect(DISTINCT block.uuid) AS supporting_block_uuids
}
RETURN argument AS argument,
       premise_uuids,
       conclusion.uuid AS conclusion_uuid,
       supporting_block_uuids
ORDER BY argument.created_at ASC, argument.uuid ASC
"""
