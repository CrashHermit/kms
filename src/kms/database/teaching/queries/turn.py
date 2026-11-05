"""Cypher statements for immutable teaching turns and state commits."""

CHECK_TEACHING_CITATIONS = """
MATCH (user:User {uuid: $user_uuid})
WITH user,
     $citation_block_uuids AS requested_block_uuids,
     $citation_image_uuids AS requested_image_uuids,
     $citation_proposition_uuids AS requested_proposition_uuids,
     $supporting_block_uuids AS requested_support_block_uuids
CALL (user, requested_block_uuids) {
    UNWIND CASE WHEN size(requested_block_uuids) = 0
                THEN [null] ELSE requested_block_uuids END AS block_uuid
    OPTIONAL MATCH (user)-[:OWNS_SOURCE]->(:Source)-[:HAS_PAGE]->
        (:SourcePage)-[:CONTAINS_BLOCK]->(
            block:SourceBlock {uuid: block_uuid}
        )
    RETURN collect(DISTINCT block.uuid) AS owned_block_uuids
}
CALL (user, requested_support_block_uuids) {
    UNWIND CASE WHEN size(requested_support_block_uuids) = 0
                THEN [null] ELSE requested_support_block_uuids END AS block_uuid
    OPTIONAL MATCH (user)-[:OWNS_SOURCE]->(:Source)-[:HAS_PAGE]->
        (:SourcePage)-[:CONTAINS_BLOCK]->(
            block:SourceBlock {uuid: block_uuid}
        )
    RETURN collect(DISTINCT block.uuid) AS owned_support_block_uuids
}
CALL (user, requested_image_uuids) {
    UNWIND CASE WHEN size(requested_image_uuids) = 0
                THEN [null] ELSE requested_image_uuids END AS image_uuid
    OPTIONAL MATCH (user)-[:OWNS_SOURCE]->(:Source)-[:HAS_PAGE]->
        (:SourcePage)-[:CONTAINS_BLOCK]->(:SourceBlock)
        -[:CONTAINS_VISUAL_ASSET]->(
            image:VisualAsset {uuid: image_uuid}
        )
    RETURN collect(DISTINCT image.uuid) AS owned_image_uuids
}
CALL (user, requested_proposition_uuids) {
    UNWIND CASE WHEN size(requested_proposition_uuids) = 0
                THEN [null] ELSE requested_proposition_uuids END
        AS proposition_uuid
    OPTIONAL MATCH (user)-[:HAS_PROPOSITION]->(
        proposition:Proposition {uuid: proposition_uuid}
    )
    RETURN collect(DISTINCT proposition.uuid) AS owned_proposition_uuids
}
CALL (user, requested_block_uuids) {
    UNWIND CASE WHEN size(requested_block_uuids) = 0
                THEN [null] ELSE requested_block_uuids END AS block_uuid
    OPTIONAL MATCH (user)-[:OWNS_SOURCE]->(source:Source)-[:HAS_PAGE]->
        (:SourcePage)-[:CONTAINS_BLOCK]->(
            :SourceBlock {uuid: block_uuid}
        )
    RETURN [item IN collect(DISTINCT CASE
        WHEN source IS NULL THEN null
        ELSE {source_uuid: source.uuid, revision: coalesce(source.revision, 0)}
    END) WHERE item IS NOT NULL] AS block_source_revisions
}
CALL (user, requested_image_uuids) {
    UNWIND CASE WHEN size(requested_image_uuids) = 0
                THEN [null] ELSE requested_image_uuids END AS image_uuid
    OPTIONAL MATCH (user)-[:OWNS_SOURCE]->(source:Source)-[:HAS_PAGE]->
        (:SourcePage)-[:CONTAINS_BLOCK]->(:SourceBlock)
        -[:CONTAINS_VISUAL_ASSET]->(:VisualAsset {uuid: image_uuid})
    RETURN [item IN collect(DISTINCT CASE
        WHEN source IS NULL THEN null
        ELSE {source_uuid: source.uuid, revision: coalesce(source.revision, 0)}
    END) WHERE item IS NOT NULL] AS image_source_revisions
}
CALL (user, requested_support_block_uuids) {
    UNWIND CASE WHEN size(requested_support_block_uuids) = 0
                THEN [null] ELSE requested_support_block_uuids END AS block_uuid
    OPTIONAL MATCH (user)-[:OWNS_SOURCE]->(source:Source)-[:HAS_PAGE]->
        (:SourcePage)-[:CONTAINS_BLOCK]->(
            :SourceBlock {uuid: block_uuid}
        )
    RETURN [item IN collect(DISTINCT CASE
        WHEN source IS NULL THEN null
        ELSE {source_uuid: source.uuid, revision: coalesce(source.revision, 0)}
    END) WHERE item IS NOT NULL] AS support_source_revisions
}
WITH requested_block_uuids, requested_image_uuids,
     requested_proposition_uuids, requested_support_block_uuids,
     owned_block_uuids, owned_image_uuids, owned_proposition_uuids,
     owned_support_block_uuids,
     block_source_revisions + image_source_revisions
         + support_source_revisions AS required_revisions
WHERE size(owned_block_uuids) = size(requested_block_uuids)
  AND size(owned_image_uuids) = size(requested_image_uuids)
  AND size(owned_proposition_uuids) = size(requested_proposition_uuids)
  AND size(owned_support_block_uuids) = size(requested_support_block_uuids)
  AND all(required IN required_revisions WHERE any(expected IN $source_revisions
      WHERE expected.source_uuid = required.source_uuid
        AND expected.revision = required.revision))
RETURN true AS valid
"""

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
WITH session
OPTIONAL MATCH (session)-[old_last:LAST_TURN]->(last_turn:TeachingTurn)
DELETE old_last
CREATE (turn:TeachingTurn {
    uuid: $turn_uuid,
    role: 'teacher',
    content: $content,
    segments_json: $segments_json,
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
WITH DISTINCT session, turn
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
CALL (turn) {
    UNWIND CASE WHEN size($citation_block_uuids) = 0
                THEN [null] ELSE $citation_block_uuids END AS block_uuid
    OPTIONAL MATCH (:User {uuid: $user_uuid})-[:OWNS_SOURCE]->
        (:Source)-[:HAS_PAGE]->(:SourcePage)-[:CONTAINS_BLOCK]->
        (block:SourceBlock {uuid: block_uuid})
    WITH DISTINCT turn, block
    FOREACH (_ IN CASE WHEN block IS NULL THEN [] ELSE [1] END |
        CREATE (turn)-[:CITES_BLOCK]->(block)
    )
    RETURN count(*) AS citation_blocks_written
}
CALL (turn) {
    UNWIND CASE WHEN size($citation_image_uuids) = 0
                THEN [null] ELSE $citation_image_uuids END AS image_uuid
    OPTIONAL MATCH (:User {uuid: $user_uuid})-[:OWNS_SOURCE]->
        (:Source)-[:HAS_PAGE]->(:SourcePage)-[:CONTAINS_BLOCK]->
        (:SourceBlock)-[:CONTAINS_VISUAL_ASSET]->
        (image:VisualAsset {uuid: image_uuid})
    WITH DISTINCT turn, image
    FOREACH (_ IN CASE WHEN image IS NULL THEN [] ELSE [1] END |
        CREATE (turn)-[:CITES_IMAGE]->(image)
    )
    RETURN count(*) AS citation_images_written
}
CALL (turn) {
    UNWIND CASE WHEN size($citation_proposition_uuids) = 0
                THEN [null] ELSE $citation_proposition_uuids END AS proposition_uuid
    OPTIONAL MATCH (:User {uuid: $user_uuid})-[:HAS_PROPOSITION]->
        (proposition:Proposition {uuid: proposition_uuid})
    WITH DISTINCT turn, proposition
    FOREACH (_ IN CASE
        WHEN proposition IS NULL THEN []
        ELSE [1]
    END | CREATE (turn)-[:CITES_PROPOSITION]->(proposition))
    RETURN count(*) AS citation_propositions_written
}
WITH session, turn
SET session.status = 'awaiting_learner',
    session.phase = $phase,
    session.updated_at = $created_at,
    session.revision = $new_revision,
    session.workspace_json = CASE
        WHEN $workspace_json IS NULL THEN session.workspace_json
        ELSE $workspace_json
    END
RETURN turn.uuid AS uuid
"""
