"""Cypher statements for teaching-session lifecycle and reads."""

CREATE_TEACHING_SESSION = """
MATCH (user:User {uuid: $user_uuid})
CREATE (user)-[:HAS_TEACHING_SESSION]->(session:TeachingSession {
    uuid: $session_uuid,
    goal: $goal,
    status: $status,
    phase: $phase,
    revision: $revision,
    created_at: $created_at,
    updated_at: $updated_at,
    ended_at: $ended_at
})
RETURN session.uuid AS uuid
"""

READ_TEACHING_SESSIONS = """
MATCH (:User {uuid: $user_uuid})-[:HAS_TEACHING_SESSION]->
      (session:TeachingSession)
RETURN session.uuid AS uuid,
       session.goal AS goal,
       session.status AS status,
       session.phase AS phase,
       session.revision AS revision,
       session.created_at AS created_at,
       session.updated_at AS updated_at,
       session.ended_at AS ended_at
ORDER BY session.updated_at DESC, session.uuid
"""

LOCK_TEACHING_SESSION = """
MATCH (:User {uuid: $user_uuid})-[:HAS_TEACHING_SESSION]->
      (session:TeachingSession {uuid: $session_uuid})
SET session._teaching_write_lock = randomUUID()
REMOVE session._teaching_write_lock
RETURN session.uuid AS uuid
"""

READ_TEACHING_SESSION_REVISION = """
MATCH (:User {uuid: $user_uuid})-[:HAS_TEACHING_SESSION]->
      (session:TeachingSession {uuid: $session_uuid})
WHERE session.revision = $expected_revision
  AND session.status IN $allowed_statuses
RETURN session.uuid AS uuid
"""

LOAD_TEACHING_SESSION = """
MATCH (:User {uuid: $user_uuid})-[:HAS_TEACHING_SESSION]->
      (session:TeachingSession {uuid: $session_uuid})
CALL (session) {
    OPTIONAL MATCH (session)-[:FIRST_TURN]->(first:TeachingTurn)
    OPTIONAL MATCH turn_path=(first)-[:NEXT_TURN*0..]->(turn:TeachingTurn)
    WITH turn_path, turn
    ORDER BY CASE
        WHEN turn_path IS NULL THEN -1
        ELSE length(turn_path)
    END
    WITH collect(turn) AS ordered_turns
    UNWIND CASE
        WHEN size(ordered_turns) = 0 THEN [NULL]
        ELSE ordered_turns
    END AS ordered_turn
    OPTIONAL MATCH (ordered_turn)-[:ASKS]->(asked_question:TeachingQuestion)
    OPTIONAL MATCH (asked_question)-[:FOR_STEP]->(asked_step:TeachingPlanStep)
    OPTIONAL MATCH (ordered_turn)-[:RESPONDS_TO]->(responded_question:TeachingQuestion)
    OPTIONAL MATCH (ordered_turn)-[:PRESENTS_PLAN]->(presented_plan:TeachingPlan)
    OPTIONAL MATCH (ordered_turn)-[:ACCEPTS_PLAN]->(accepted_plan:TeachingPlan)
    WITH collect(
        CASE WHEN ordered_turn IS NULL THEN NULL ELSE {
            turn: {
                uuid: ordered_turn.uuid,
                role: ordered_turn.role,
                content: ordered_turn.content,
                created_at: ordered_turn.created_at
            },
            question: CASE WHEN asked_question IS NULL THEN NULL ELSE {
                uuid: asked_question.uuid,
                kind: asked_question.kind,
                prompt: asked_question.prompt,
                response_json: asked_question.response_json,
                created_at: asked_question.created_at
            } END,
            question_step_uuid: asked_step.uuid,
            responds_to_question_uuid: responded_question.uuid,
            presented_plan_uuid: presented_plan.uuid,
            accepted_plan_uuid: accepted_plan.uuid
        } END
    ) AS raw_turns
    RETURN [turn_row IN raw_turns WHERE turn_row IS NOT NULL] AS turns
}
CALL (session) {
    OPTIONAL MATCH (session)-[:CURRENT_ASSESSMENT]->
        (assessment:ProbeAssessment)
    RETURN CASE WHEN assessment IS NULL THEN NULL ELSE {
        uuid: assessment.uuid,
        summary: assessment.summary,
        strands_json: assessment.strands_json,
        created_at: assessment.created_at
    } END AS current_assessment
}
CALL (session) {
    OPTIONAL MATCH (session)-[:CURRENT_PLAN]->(current_plan:TeachingPlan)
    CALL (current_plan) {
        WITH current_plan AS plan_for_steps
        OPTIONAL MATCH (plan_for_steps)-[:HAS_STEP]->(step:TeachingPlanStep)
        WITH plan_for_steps, collect(
            CASE WHEN step IS NULL THEN NULL ELSE {
                uuid: step.uuid,
                title: step.title,
                objective: step.objective
            } END
        ) AS raw_steps
        OPTIONAL MATCH (plan_for_steps)-[:HAS_STEP]->
            (dependent:TeachingPlanStep)-[:REQUIRES]->
            (prerequisite:TeachingPlanStep)
        WITH plan_for_steps,
             raw_steps,
             collect(
                 CASE WHEN dependent IS NULL THEN NULL ELSE {
                     step_uuid: dependent.uuid,
                     prerequisite_uuid: prerequisite.uuid
                 } END
             ) AS raw_dependencies
        RETURN plan_for_steps,
               [step_row IN raw_steps WHERE step_row IS NOT NULL] AS steps,
               [dependency IN raw_dependencies
                WHERE dependency IS NOT NULL] AS dependencies
    }
    RETURN CASE WHEN current_plan IS NULL THEN NULL ELSE {
        uuid: current_plan.uuid,
        summary: current_plan.summary,
        created_at: current_plan.created_at,
        steps: steps,
        dependencies: dependencies
    } END AS current_plan
}
CALL (session) {
    OPTIONAL MATCH (session)-[:CURRENT_PLAN]->(current_plan:TeachingPlan)
    OPTIONAL MATCH (current_plan)<-[:ACCEPTS_PLAN]->
        (accepting_turn:TeachingTurn {role: 'learner'})
    RETURN CASE
        WHEN current_plan IS NULL OR accepting_turn IS NULL THEN NULL
        ELSE {
            plan_uuid: current_plan.uuid,
            learner_turn_uuid: accepting_turn.uuid
        }
    END AS current_plan_acceptance
}
CALL (session) {
    OPTIONAL MATCH (session)-[:CURRENT_STEP]->(current_step:TeachingPlanStep)
    RETURN current_step.uuid AS current_step_uuid
}
CALL (session) {
    OPTIONAL MATCH (session)-[:PENDING_QUESTION]->
        (pending_question:TeachingQuestion)
    RETURN pending_question.uuid AS pending_question_uuid
}
RETURN session.uuid AS uuid,
       session.goal AS goal,
       session.status AS status,
       session.phase AS phase,
       session.revision AS revision,
       session.created_at AS created_at,
       session.updated_at AS updated_at,
       session.ended_at AS ended_at,
       turns,
       current_assessment,
       current_plan,
       current_plan_acceptance,
       current_step_uuid,
       pending_question_uuid
"""

END_TEACHING_SESSION = """
MATCH (:User {uuid: $user_uuid})-[:HAS_TEACHING_SESSION]->
      (session:TeachingSession {uuid: $session_uuid})
OPTIONAL MATCH (session)-[pending:PENDING_QUESTION]->()
DELETE pending
SET session.status = 'ended',
    session.updated_at = $ended_at,
    session.ended_at = $ended_at,
    session.revision = $new_revision
RETURN session.uuid AS uuid
"""
