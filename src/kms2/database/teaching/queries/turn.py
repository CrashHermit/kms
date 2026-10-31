"""Cypher statements for immutable teaching turns and state commits."""

APPEND_TEACHING_RESPONSE = """
MATCH (:User {uuid: $user_uuid})-[:HAS_TEACHING_SESSION]->
      (session:TeachingSession {uuid: $session_uuid})
OPTIONAL MATCH (session)-[old_last:LAST_TURN]->(last_turn:TeachingTurn)
DELETE old_last
CREATE (turn:TeachingTurn {
    uuid: $turn_uuid,
    role: 'learner',
    content: $content,
    created_at: $created_at
})
CREATE (session)-[:HAS_TURN]->(turn)
FOREACH (_ IN CASE
    WHEN last_turn IS NULL THEN []
    ELSE [1]
END | CREATE (last_turn)-[:NEXT_TURN]->(turn))
FOREACH (_ IN CASE
    WHEN last_turn IS NULL THEN [1]
    ELSE []
END | CREATE (session)-[:FIRST_TURN]->(turn))
CREATE (session)-[:LAST_TURN]->(turn)
WITH session, turn
OPTIONAL MATCH (session)-[:PENDING_QUESTION]->(pending_question:TeachingQuestion)
FOREACH (question IN CASE
    WHEN pending_question IS NULL THEN []
    ELSE [pending_question]
END | CREATE (turn)-[:RESPONDS_TO]->(question))
SET session.status = 'awaiting_teacher',
    session.updated_at = $created_at,
    session.revision = $new_revision
RETURN turn.uuid AS uuid
"""

RECORD_TEACHING_TURN = """
MATCH (:User {uuid: $user_uuid})-[:HAS_TEACHING_SESSION]->
      (session:TeachingSession {uuid: $session_uuid})
OPTIONAL MATCH (session)-[old_last:LAST_TURN]->(last_turn:TeachingTurn)
DELETE old_last
CREATE (turn:TeachingTurn {
    uuid: $turn_uuid,
    role: 'teacher',
    content: $content,
    created_at: $created_at
})
CREATE (session)-[:HAS_TURN]->(turn)
FOREACH (_ IN CASE
    WHEN last_turn IS NULL THEN []
    ELSE [1]
END | CREATE (last_turn)-[:NEXT_TURN]->(turn))
FOREACH (_ IN CASE
    WHEN last_turn IS NULL THEN [1]
    ELSE []
END | CREATE (session)-[:FIRST_TURN]->(turn))
CREATE (session)-[:LAST_TURN]->(turn)
FOREACH (_ IN CASE
    WHEN $new_assessment_uuid IS NULL THEN []
    ELSE [1]
END |
    CREATE (assessment:ProbeAssessment {
        uuid: $new_assessment_uuid,
        summary: $new_assessment_summary,
        strands_json: $new_assessment_strands_json,
        created_at: $new_assessment_created_at
    })
    CREATE (session)-[:HAS_ASSESSMENT]->(assessment)
)
FOREACH (_ IN CASE
    WHEN $new_plan_uuid IS NULL THEN []
    ELSE [1]
END |
    CREATE (plan:TeachingPlan {
        uuid: $new_plan_uuid,
        summary: $new_plan_summary,
        created_at: $new_plan_created_at
    })
    CREATE (session)-[:HAS_PLAN]->(plan)
)
WITH session, turn
OPTIONAL MATCH (session)-[:HAS_PLAN]->(new_plan:TeachingPlan {
    uuid: $new_plan_uuid
})
FOREACH (step_data IN $new_plan_steps |
    CREATE (new_step:TeachingPlanStep {
        uuid: step_data.uuid,
        title: step_data.title,
        objective: step_data.objective
    })
    CREATE (new_plan)-[:HAS_STEP]->(new_step)
)
WITH session, turn
OPTIONAL MATCH (session)-[:HAS_PLAN]->(new_plan:TeachingPlan {
    uuid: $new_plan_uuid
})
UNWIND CASE
    WHEN size($new_plan_dependencies) = 0 THEN [NULL]
    ELSE $new_plan_dependencies
END AS dependency_data
OPTIONAL MATCH (new_plan)-[:HAS_STEP]->
    (dependent_step:TeachingPlanStep {uuid: dependency_data.step_uuid})
OPTIONAL MATCH (new_plan)-[:HAS_STEP]->
    (prerequisite_step:TeachingPlanStep {
        uuid: dependency_data.prerequisite_uuid
    })
FOREACH (_ IN CASE
    WHEN dependency_data IS NULL THEN []
    ELSE [1]
END |
    CREATE (dependent_step)-[:REQUIRES]->(prerequisite_step)
)
WITH session, turn
FOREACH (_ IN CASE
    WHEN $new_question_uuid IS NULL THEN []
    ELSE [1]
END |
    CREATE (question:TeachingQuestion {
        uuid: $new_question_uuid,
        kind: $new_question_kind,
        prompt: $new_question_prompt,
        response_json: $new_question_response_json,
        created_at: $new_question_created_at
    })
    CREATE (turn)-[:ASKS]->(question)
)
WITH session, turn
OPTIONAL MATCH (turn)-[:ASKS]->(new_question:TeachingQuestion)
OPTIONAL MATCH (session)-[:HAS_PLAN]->(:TeachingPlan)-[:HAS_STEP]->
    (question_step:TeachingPlanStep {uuid: $new_question_step_uuid})
FOREACH (_ IN CASE
    WHEN new_question IS NULL OR question_step IS NULL THEN []
    ELSE [1]
END |
    CREATE (new_question)-[:FOR_STEP]->(question_step)
)
WITH session, turn
OPTIONAL MATCH (session)-[:HAS_PLAN]->
    (presented_plan:TeachingPlan {uuid: $new_plan_uuid})
FOREACH (_ IN CASE
    WHEN presented_plan IS NULL THEN []
    ELSE [1]
END |
    CREATE (turn)-[:PRESENTS_PLAN]->(presented_plan)
)
WITH session, turn
OPTIONAL MATCH (session)-[old_assessment:CURRENT_ASSESSMENT]->()
DELETE old_assessment
WITH session, turn
OPTIONAL MATCH (session)-[old_plan:CURRENT_PLAN]->()
DELETE old_plan
WITH session, turn
OPTIONAL MATCH (session)-[old_step:CURRENT_STEP]->()
DELETE old_step
WITH session, turn
OPTIONAL MATCH (session)-[old_pending:PENDING_QUESTION]->()
DELETE old_pending
WITH session, turn
OPTIONAL MATCH (session)-[:HAS_ASSESSMENT]->
    (selected_assessment:ProbeAssessment {
        uuid: $current_assessment_uuid
    })
FOREACH (_ IN CASE
    WHEN selected_assessment IS NULL THEN []
    ELSE [1]
END |
    CREATE (session)-[:CURRENT_ASSESSMENT]->(selected_assessment)
)
WITH session, turn
OPTIONAL MATCH (session)-[:HAS_PLAN]->
    (selected_plan:TeachingPlan {uuid: $current_plan_uuid})
FOREACH (_ IN CASE
    WHEN selected_plan IS NULL THEN []
    ELSE [1]
END |
    CREATE (session)-[:CURRENT_PLAN]->(selected_plan)
)
WITH session, turn
OPTIONAL MATCH (session)-[:HAS_PLAN]->
    (:TeachingPlan {uuid: $current_plan_uuid})-[:HAS_STEP]->
    (selected_step:TeachingPlanStep {uuid: $current_step_uuid})
FOREACH (_ IN CASE
    WHEN selected_step IS NULL THEN []
    ELSE [1]
END |
    CREATE (session)-[:CURRENT_STEP]->(selected_step)
)
WITH session, turn
OPTIONAL MATCH (session)-[:HAS_TURN]->(:TeachingTurn)-[:ASKS]->
    (selected_question:TeachingQuestion {uuid: $pending_question_uuid})
FOREACH (_ IN CASE
    WHEN selected_question IS NULL THEN []
    ELSE [1]
END |
    CREATE (session)-[:PENDING_QUESTION]->(selected_question)
)
WITH session, turn
OPTIONAL MATCH (session)-[:CURRENT_PLAN]->
    (accepted_plan:TeachingPlan {uuid: $acceptance_plan_uuid})
OPTIONAL MATCH (session)-[:HAS_TURN]->
    (accepting_turn:TeachingTurn {
        uuid: $acceptance_learner_turn_uuid,
        role: 'learner'
    })
FOREACH (_ IN CASE
    WHEN accepted_plan IS NULL OR accepting_turn IS NULL THEN []
    ELSE [1]
END |
    CREATE (accepting_turn)-[:ACCEPTS_PLAN]->(accepted_plan)
)
WITH session, turn
SET session.status = 'awaiting_learner',
    session.phase = $phase,
    session.updated_at = $created_at,
    session.revision = $new_revision
RETURN turn.uuid AS uuid
"""
