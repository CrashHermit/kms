"""Owner-scoped learner observation search, reads, and provenance writes."""

SEARCH_OWNED_LEARNER_OBSERVATIONS = """
MATCH (user:User {uuid: $user_uuid})-[:HAS_LEARNER_OBSERVATION]->
      (observation:LearnerObservation)
WHERE observation.embedding IS NOT NULL
WITH user, observation,
     vector.similarity.cosine(observation.embedding, $query_embedding)
         AS similarity
ORDER BY similarity DESC, observation.created_at DESC, observation.uuid ASC
LIMIT $candidate_limit
CALL (user, observation) {
    MATCH (observation)-[:EVIDENCED_BY]->(learner:TeachingTurn {role: 'learner'})
    MATCH (user)-[:HAS_TEACHING_SESSION]->
          (learner_session:TeachingSession)-[:HAS_TURN]->(learner)
    CALL (learner) {
        OPTIONAL MATCH (learner)-[:RESPONDS_TO]->(question:TeachingQuestion)
        RETURN CASE WHEN question IS NULL THEN NULL ELSE {
            uuid: question.uuid,
            kind: question.kind,
            prompt: question.prompt,
            response_json: question.response_json,
            created_at: question.created_at
        } END AS question
    }
    WITH learner_session, learner, question
    ORDER BY learner.created_at ASC, learner.uuid ASC
    RETURN collect({
        turn_uuid: learner.uuid,
        session_uuid: learner_session.uuid,
        content: learner.content,
        created_at: learner.created_at,
        question: question
    }) AS evidence
}
WITH observation, similarity, evidence
ORDER BY similarity DESC, observation.created_at DESC, observation.uuid ASC
RETURN {
    observation: {
        uuid: observation.uuid,
        kind: observation.kind,
        topic: observation.topic,
        description: observation.description,
        embedding: observation.embedding,
        created_at: observation.created_at
    },
    evidence: evidence
} AS record,
similarity
"""

READ_OWNED_LEARNER_OBSERVATIONS = """
UNWIND range(0, size($observation_uuids) - 1) AS position
WITH position, $observation_uuids[position] AS observation_uuid
MATCH (user:User {uuid: $user_uuid})-[:HAS_LEARNER_OBSERVATION]->
      (observation:LearnerObservation {uuid: observation_uuid})
CALL (user, observation) {
    MATCH (observation)-[:EVIDENCED_BY]->(learner:TeachingTurn {role: 'learner'})
    MATCH (user)-[:HAS_TEACHING_SESSION]->
          (learner_session:TeachingSession)-[:HAS_TURN]->(learner)
    CALL (learner) {
        OPTIONAL MATCH (learner)-[:RESPONDS_TO]->(question:TeachingQuestion)
        RETURN CASE WHEN question IS NULL THEN NULL ELSE {
            uuid: question.uuid,
            kind: question.kind,
            prompt: question.prompt,
            response_json: question.response_json,
            created_at: question.created_at
        } END AS question
    }
    WITH learner_session, learner, question
    ORDER BY learner.created_at ASC, learner.uuid ASC
    RETURN collect({
        turn_uuid: learner.uuid,
        session_uuid: learner_session.uuid,
        content: learner.content,
        created_at: learner.created_at,
        question: question
    }) AS evidence
}
RETURN position, {
    observation: {
        uuid: observation.uuid,
        kind: observation.kind,
        topic: observation.topic,
        description: observation.description,
        embedding: observation.embedding,
        created_at: observation.created_at
    },
    evidence: evidence
} AS record
ORDER BY position
"""

AUTHORIZE_OWNED_LEARNER_OBSERVATIONS = """
MATCH (user:User {uuid: $user_uuid})
UNWIND $observation_uuids AS observation_uuid
OPTIONAL MATCH (user)-[:HAS_LEARNER_OBSERVATION]->
    (observation:LearnerObservation {uuid: observation_uuid})
WITH user, collect(DISTINCT observation.uuid) AS owned_uuids
WHERE size(owned_uuids) = size($observation_uuids)
RETURN user.uuid AS uuid
"""

LINK_ASSESSMENT_OBSERVATIONS = """
MATCH (user:User {uuid: $user_uuid})
UNWIND $observation_uuids AS observation_uuid
MATCH (user)-[:HAS_LEARNER_OBSERVATION]->
    (observation:LearnerObservation {uuid: observation_uuid})
WITH user, collect(DISTINCT observation) AS observations
WHERE size(observations) = size($observation_uuids)
MATCH (user)-[:HAS_TEACHING_SESSION]->
    (session:TeachingSession {uuid: $session_uuid})-[:HAS_ASSESSMENT]->
    (assessment:ProbeAssessment {uuid: $assessment_uuid})
FOREACH (observation IN observations |
    CREATE (assessment)-[:BASED_ON]->(observation)
)
RETURN assessment.uuid AS uuid
"""

WRITE_LEARNER_OBSERVATION = """
MATCH (user:User {uuid: $user_uuid})-[:HAS_TEACHING_SESSION]->
      (session:TeachingSession)-[:HAS_TURN]->
      (teacher:TeachingTurn {uuid: $teacher_turn_uuid, role: 'teacher'})
WITH user, teacher
UNWIND $evidence_turn_uuids AS evidence_uuid
MATCH (user)-[:HAS_TEACHING_SESSION]->
      (evidence_session:TeachingSession)-[:HAS_TURN]->
      (learner:TeachingTurn {uuid: evidence_uuid, role: 'learner'})
WITH user, teacher, collect(DISTINCT learner) AS learners
WHERE size(learners) = size($evidence_turn_uuids)
CREATE (observation:LearnerObservation {
    uuid: $uuid,
    kind: $kind,
    topic: $topic,
    description: $description,
    embedding: $embedding,
    created_at: $created_at
})
CREATE (user)-[:HAS_LEARNER_OBSERVATION]->(observation)
CREATE (teacher)-[:RECORDED_OBSERVATION]->(observation)
FOREACH (learner IN learners |
    CREATE (observation)-[:EVIDENCED_BY]->(learner)
)
RETURN observation.uuid AS uuid
"""
