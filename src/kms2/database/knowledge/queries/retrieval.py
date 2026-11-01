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
        OPTIONAL MATCH (assertion)-[:DERIVED_BY]->(step:ReasoningStep)
        WHERE (user)-[:HAS_REASONING_STEP]->(step)
        RETURN collect(DISTINCT step.uuid) AS reasoning_step_uuids
    }
    RETURN [
        item IN collect(CASE WHEN assertion IS NULL THEN NULL ELSE {
            assertion: assertion,
            proposition_uuid: proposition.uuid,
            supporting_block_uuids: supporting_block_uuids,
            reasoning_step_uuids: reasoning_step_uuids,
            teaching_turn_uuid: turn.uuid
        } END)
        WHERE item IS NOT NULL
    ] AS assertions
}
CALL (user, proposition) {
    OPTIONAL MATCH (user)-[:HAS_REASONING_STEP]->
          (step:ReasoningStep)-[:HAS_CONCLUSION]->(proposition)
    RETURN collect(DISTINCT step.uuid) AS reasoning_step_uuids
}
RETURN proposition AS proposition, assertions, reasoning_step_uuids
"""

READ_OWNED_REASONING_STEP = """
MATCH (user:User {uuid: $user_uuid})-[:HAS_REASONING_STEP]->(
    step:ReasoningStep {uuid: $reasoning_step_uuid}
)
MATCH (user)-[:HAS_PROPOSITION]->(conclusion:Proposition)
      <-[:HAS_CONCLUSION]-(step)
CALL (user, step) {
    OPTIONAL MATCH (step)-[:HAS_PREMISE]->(premise:Proposition)
    WHERE (user)-[:HAS_PROPOSITION]->(premise)
    RETURN collect(DISTINCT premise.uuid) AS premise_uuids
}
CALL (user, step) {
    OPTIONAL MATCH (step)-[:SUPPORTED_BY]->(block:SourceBlock)
    WHERE EXISTS {
        MATCH (user)-[:OWNS_SOURCE]->(:Source)-[:HAS_PAGE]->
              (:SourcePage)-[:CONTAINS_BLOCK]->(block)
    }
    RETURN collect(DISTINCT block.uuid) AS supporting_block_uuids
}
RETURN step AS step,
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

FIND_OWNED_REASONING_STEP_CANDIDATES = """
MATCH (user:User {uuid: $user_uuid})-[:HAS_REASONING_STEP]->(
    step:ReasoningStep
)
MATCH (user)-[:HAS_PROPOSITION]->(conclusion:Proposition)
      <-[:HAS_CONCLUSION]-(step)
WHERE conclusion.uuid = $conclusion_uuid
  AND step.explanation = $explanation
CALL (user, step) {
    OPTIONAL MATCH (step)-[:HAS_PREMISE]->(premise:Proposition)
    WHERE (user)-[:HAS_PROPOSITION]->(premise)
    RETURN collect(DISTINCT premise.uuid) AS premise_uuids
}
CALL (user, step) {
    OPTIONAL MATCH (step)-[:SUPPORTED_BY]->(block:SourceBlock)
    WHERE EXISTS {
        MATCH (user)-[:OWNS_SOURCE]->(:Source)-[:HAS_PAGE]->
              (:SourcePage)-[:CONTAINS_BLOCK]->(block)
    }
    RETURN collect(DISTINCT block.uuid) AS supporting_block_uuids
}
RETURN step AS step,
       premise_uuids,
       conclusion.uuid AS conclusion_uuid,
       supporting_block_uuids
ORDER BY step.created_at ASC, step.uuid ASC
"""
