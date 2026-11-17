"""Cypher statements for owner-scoped FSRS memory and review records."""

LOCK_MEMORY_OWNER = """
MATCH (user:User {uuid: $user_uuid})
SET user._memory_lock = randomUUID()
REMOVE user._memory_lock
RETURN user.uuid AS uuid
"""

REQUIRE_MEMORY_OWNER = """
MATCH (user:User {uuid: $user_uuid})
RETURN user.uuid AS uuid
"""

READ_REVIEW_INPUT = """
MATCH (user:User {uuid: $user_uuid})-[:OWNS_SESSION]->(session:LearningSession {uuid: $session_uuid})
MATCH (session)-[:HAS_STEP]->(task:Task)-[:HAS_RESPONSE]->(response:TaskResponse)
MATCH (response)-[:HAS_ASSESSMENT]->(assessment:TaskAssessment {uuid: $assessment_uuid})
MATCH (task)-[:ASSESSES]->(target)
CALL (user, target) {
    MATCH (user)-[:OWNS_PROPOSITION]->(owned_target:Proposition)
    WHERE target:Proposition AND owned_target.uuid = target.uuid
    RETURN owned_target
    UNION
    MATCH (user)-[:OWNS_ARGUMENT]->(owned_target:Argument)
    WHERE target:Argument AND owned_target.uuid = target.uuid
    RETURN owned_target
    UNION
    MATCH (user)-[:OWNS_SESSION]->(:LearningSession)-[:HAS_STEP]->(:Task)-[:HAS_SOLUTION]->(owned_target:Solution)
    WHERE target:Solution AND owned_target.uuid = target.uuid
    RETURN owned_target
}
OPTIONAL MATCH (user)-[:HAS_MEMORY]->(memory:LearningMemory)-[:FOR_TARGET]->(owned_target)
RETURN owned_target {
    .uuid,
    kind: CASE WHEN owned_target:Proposition THEN 'proposition'
               WHEN owned_target:Argument THEN 'argument'
               ELSE 'solution' END
} AS target,
       response.uuid AS response_uuid,
       assessment.uuid AS assessment_uuid,
       assessment.payload_json AS assessment_payload_json,
       response.created_at AS reviewed_at,
       memory {
           .uuid, .card_json, .due_at, .review_count,
           .created_at, .updated_at, .invalidated_at
       } AS memory
"""

SAVE_REVIEW = """
MATCH (user:User {uuid: $user_uuid})
CALL (user) {
    MATCH (user)-[:OWNS_PROPOSITION]->(target:Proposition {uuid: $target_uuid})
    WHERE $target_kind = 'proposition'
    RETURN target
    UNION
    MATCH (user)-[:OWNS_ARGUMENT]->(target:Argument {uuid: $target_uuid})
    WHERE $target_kind = 'argument'
    RETURN target
    UNION
    MATCH (user)-[:OWNS_SESSION]->(:LearningSession)-[:HAS_STEP]->(:Task)-[:HAS_SOLUTION]->(target:Solution {uuid: $target_uuid})
    WHERE $target_kind = 'solution'
    RETURN target
}
MERGE (user)-[owner_edge:HAS_MEMORY]->(memory:LearningMemory {uuid: $memory_uuid})
ON CREATE SET owner_edge.created_at = $created_at,
              owner_edge.updated_at = $created_at,
              memory.created_at = $memory_created_at,
              memory.invalidated_at = null
SET owner_edge.updated_at = $updated_at,
    memory.card_json = $card_json,
    memory.due_at = $due_at,
    memory.review_count = $review_count,
    memory.updated_at = $updated_at
MERGE (memory)-[target_edge:FOR_TARGET]->(target)
ON CREATE SET target_edge.created_at = $created_at,
              target_edge.updated_at = $created_at
CREATE (review:MemoryReview {
    uuid: $review_uuid, rating: $rating, reviewed_at: $reviewed_at,
    sequence: $sequence, created_at: $created_at,
    updated_at: $updated_at, invalidated_at: null
})
CREATE (memory)-[:HAS_REVIEW {created_at: $created_at, updated_at: $created_at}]->(review)
WITH user, memory, review, target
MATCH (user)-[:OWNS_SESSION]->(:LearningSession)-[:HAS_STEP]->(task:Task)-[:HAS_RESPONSE]->(response:TaskResponse)
MATCH (response)-[:HAS_ASSESSMENT]->(assessment:TaskAssessment {uuid: $assessment_uuid})
MATCH (task)-[:ASSESSES]->(target)
CREATE (review)-[:BASED_ON {created_at: $created_at, updated_at: $created_at}]->(assessment)
RETURN memory.uuid AS uuid
"""

READ_MEMORY = """
MATCH (:User {uuid: $user_uuid})-[:HAS_MEMORY]->(memory:LearningMemory)-[:FOR_TARGET]->(target)
WHERE target.uuid = $target_uuid AND
      (($target_kind = 'proposition' AND target:Proposition) OR
       ($target_kind = 'argument' AND target:Argument) OR
       ($target_kind = 'solution' AND target:Solution))
CALL (memory) {
    OPTIONAL MATCH (memory)-[:HAS_REVIEW]->(review:MemoryReview)-[:BASED_ON]->(assessment:TaskAssessment)
    RETURN collect({
        review: review {
            .uuid, .rating, .reviewed_at, .sequence,
            .created_at, .updated_at, .invalidated_at
        },
        assessment_uuid: assessment.uuid
    }) AS reviews
}
RETURN memory {
    .uuid, .card_json, .due_at, .review_count,
    .created_at, .updated_at, .invalidated_at
} AS memory,
       target {
           .uuid,
           kind: CASE WHEN target:Proposition THEN 'proposition'
                      WHEN target:Argument THEN 'argument'
                      ELSE 'solution' END
       } AS target,
       reviews
"""

LIST_DUE = """
MATCH (:User {uuid: $user_uuid})-[:HAS_MEMORY]->(memory:LearningMemory)-[:FOR_TARGET]->(target)
WHERE memory.due_at <= $as_of
RETURN memory {
    .uuid, .card_json, .due_at, .review_count,
    .created_at, .updated_at, .invalidated_at
} AS memory,
       target {
           .uuid,
           kind: CASE WHEN target:Proposition THEN 'proposition'
                      WHEN target:Argument THEN 'argument'
                      ELSE 'solution' END
       } AS target
ORDER BY memory.due_at, target.kind, target.uuid
"""

READ_TARGET_PROPOSITION = """
MATCH (:User {uuid: $user_uuid})-[:OWNS_PROPOSITION]->(proposition:Proposition {uuid: $target_uuid})
RETURN proposition {
    .uuid, .name, .statement, .created_at, .updated_at, .invalidated_at
} AS proposition
"""

READ_TARGET_ARGUMENT = """
MATCH (:User {uuid: $user_uuid})-[:OWNS_ARGUMENT]->(argument:Argument {uuid: $target_uuid})
MATCH (argument)-[:HAS_PREMISE]->(premise:Proposition)
MATCH (argument)-[:CONCLUDES]->(conclusion:Proposition)
MATCH (:User {uuid: $user_uuid})-[:OWNS_PROPOSITION]->(premise)
MATCH (:User {uuid: $user_uuid})-[:OWNS_PROPOSITION]->(conclusion)
RETURN argument {
    .uuid, .created_at, .updated_at, .invalidated_at
} AS argument,
collect(DISTINCT premise {
    .uuid, .name, .statement, .created_at, .updated_at, .invalidated_at
}) AS premises,
conclusion {
    .uuid, .name, .statement, .created_at, .updated_at, .invalidated_at
} AS conclusion
"""

READ_TARGET_SOLUTION_TASK = """
MATCH (:User {uuid: $user_uuid})-[:OWNS_SESSION]->(session:LearningSession)-[:HAS_STEP]->(task:Task)-[:HAS_SOLUTION]->(:Solution {uuid: $target_uuid})
RETURN task.uuid AS task_uuid, session.uuid AS session_uuid
"""
