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
LOCK_TEACHING_USER_KNOWLEDGE = """
MATCH (user:User {uuid: $user_uuid})
SET user._knowledge_write_lock = randomUUID()
REMOVE user._knowledge_write_lock
RETURN user.uuid AS uuid
"""

READ_TEACHING_USER_KNOWLEDGE_REVISION = """
MATCH (user:User {uuid: $user_uuid})
WHERE coalesce(user.knowledge_revision, 0) = $expected_revision
RETURN user.uuid AS uuid
"""

LOCK_OWNED_SOURCE = """
MATCH (:User {uuid: $user_uuid})-[:OWNS_SOURCE]->
      (source:Source {uuid: $source_uuid})
SET source._source_write_lock = randomUUID()
REMOVE source._source_write_lock
RETURN source.uuid AS uuid
"""

CHECK_OWNED_SOURCE_REVISION = """
MATCH (:User {uuid: $user_uuid})-[:OWNS_SOURCE]->
      (source:Source {uuid: $source_uuid})
WHERE coalesce(source.revision, 0) = $revision
RETURN source.uuid AS uuid
"""

READ_TEACHING_SESSION_REVISION = """
MATCH (:User {uuid: $user_uuid})-[:HAS_TEACHING_SESSION]->
      (session:TeachingSession {uuid: $session_uuid})
WHERE session.revision = $expected_revision
  AND session.status IN $allowed_statuses
RETURN session.uuid AS uuid
"""

LOAD_TEACHING_SESSION = """
MATCH (user:User {uuid: $user_uuid})-[:HAS_TEACHING_SESSION]->
      (session:TeachingSession {uuid: $session_uuid})
CALL (session, user) {
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
    CALL (ordered_turn) {
        OPTIONAL MATCH (ordered_turn)-[:CITES_BLOCK]->(
            block:SourceBlock
        )
        WHERE EXISTS {
            MATCH (:User {uuid: $user_uuid})-[:OWNS_SOURCE]->(
                :Source
            )-[:HAS_PAGE]->(:SourcePage)-[:CONTAINS_BLOCK]->(block)
        }
        RETURN collect(DISTINCT CASE WHEN block IS NULL THEN null ELSE {
            kind: 'block',
            uuid: block.uuid
        } END) AS block_citations
    }
    CALL (ordered_turn) {
        OPTIONAL MATCH (ordered_turn)-[:CITES_IMAGE]->(
            image:VisualAsset
        )
        WHERE EXISTS {
            MATCH (:User {uuid: $user_uuid})-[:OWNS_SOURCE]->(
                :Source
            )-[:HAS_PAGE]->(:SourcePage)-[:CONTAINS_BLOCK]->(
                :SourceBlock
            )-[:CONTAINS_VISUAL_ASSET]->(image)
        }
        RETURN collect(DISTINCT CASE WHEN image IS NULL THEN null ELSE {
            kind: 'image',
            uuid: image.uuid
        } END) AS image_citations
    }
    CALL (ordered_turn) {
        OPTIONAL MATCH (ordered_turn)-[:CITES_PROPOSITION]->(
            proposition:Proposition
        )
        WHERE EXISTS {
            MATCH (:User {uuid: $user_uuid})-[:HAS_PROPOSITION]->(proposition)
        }
        RETURN collect(DISTINCT CASE WHEN proposition IS NULL THEN null ELSE {
            kind: 'proposition',
            uuid: proposition.uuid
        } END) AS proposition_citations
    }
    CALL (ordered_turn) {
        OPTIONAL MATCH (ordered_turn)-[:HAS_KNOWLEDGE_ASSERTION]->
            (assertion:KnowledgeAssertion)-[:ASSERTS]->(
                asserted:Proposition
            )
        WITH assertion, asserted
        ORDER BY assertion.uuid
        CALL (assertion) {
            OPTIONAL MATCH (assertion)-[:SUPPORTED_BY]->(
                support:SourceBlock
            )
            WITH DISTINCT support.uuid AS uuid
            WHERE uuid IS NOT NULL
            ORDER BY uuid
            RETURN collect(uuid) AS support_uuids
        }
        CALL (assertion) {
            OPTIONAL MATCH (assertion)-[:DERIVED_BY]->(
                argument:Argument
            )
            WITH DISTINCT argument.uuid AS uuid
            WHERE uuid IS NOT NULL
            ORDER BY uuid
            RETURN collect(uuid) AS argument_uuids
        }
        WITH assertion, asserted, support_uuids, argument_uuids
        ORDER BY assertion.uuid
        RETURN collect(CASE WHEN assertion IS NULL THEN null ELSE {
            uuid: assertion.uuid,
            created_at: assertion.created_at,
            proposition_uuid: asserted.uuid,
            supporting_block_uuids: support_uuids,
            argument_uuids: argument_uuids,
            teaching_turn_uuid: ordered_turn.uuid
        } END) AS assertion_rows
    }
    CALL (user, ordered_turn) {
        OPTIONAL MATCH (ordered_turn)-[:RECORDED_OBSERVATION]->
            (observation:LearnerObservation)
        WITH user, observation
        ORDER BY observation.created_at ASC, observation.uuid ASC
        CALL (user, observation) {
            OPTIONAL MATCH (observation)-[:EVIDENCED_BY]->
                (learner:TeachingTurn {role: 'learner'})
            OPTIONAL MATCH (user)-[:HAS_TEACHING_SESSION]->
                (evidence_session:TeachingSession)-[:HAS_TURN]->(learner)
            CALL (learner) {
                OPTIONAL MATCH (learner)-[:RESPONDS_TO]->
                    (question:TeachingQuestion)
                RETURN CASE WHEN question IS NULL THEN NULL ELSE {
                    uuid: question.uuid,
                    kind: question.kind,
                    prompt: question.prompt,
                    response_json: question.response_json,
                    created_at: question.created_at
                } END AS question
            }
            WITH evidence_session, learner, question
            ORDER BY learner.created_at ASC, learner.uuid ASC
            RETURN collect({
                turn_uuid: learner.uuid,
                session_uuid: evidence_session.uuid,
                content: learner.content,
                created_at: learner.created_at,
                question: question
            }) AS evidence
        }
        RETURN collect(CASE WHEN observation IS NULL THEN NULL ELSE {
            observation: {
                uuid: observation.uuid,
                kind: observation.kind,
                topic: observation.topic,
                description: observation.description,
                embedding: observation.embedding,
                created_at: observation.created_at
            },
            evidence: evidence
        } END) AS raw_observations
    }
    WITH ordered_turn, asked_question, asked_step, responded_question,
         presented_plan, accepted_plan, block_citations, image_citations,
         proposition_citations,
         [item IN raw_observations WHERE item IS NOT NULL]
             AS learner_observations,
         [item IN assertion_rows WHERE item IS NOT NULL] AS assertions
    WITH collect(
        CASE WHEN ordered_turn IS NULL THEN NULL ELSE {
            turn: {
                uuid: ordered_turn.uuid,
                role: ordered_turn.role,
                content: ordered_turn.content,
                segments_json: ordered_turn.segments_json,
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
            accepted_plan_uuid: accepted_plan.uuid,
            citations: block_citations + image_citations +
                proposition_citations,
            knowledge_assertions: assertions,
            learner_observations: learner_observations
        } END
    ) AS raw_turns
    RETURN [turn_row IN raw_turns WHERE turn_row IS NOT NULL] AS turns
}
CALL (session) {
    OPTIONAL MATCH (session)-[:CURRENT_ASSESSMENT]->
        (assessment:ProbeAssessment)
    CALL (assessment) {
        OPTIONAL MATCH (assessment)-[:BASED_ON]->
            (observation:LearnerObservation)
        RETURN collect(DISTINCT observation.uuid) AS observation_uuids
    }
    RETURN CASE WHEN assessment IS NULL THEN NULL ELSE {
        uuid: assessment.uuid,
        summary: assessment.summary,
        strands_json: assessment.strands_json,
        observation_uuids: observation_uuids,
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
       session.workspace_json AS workspace_json,
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
