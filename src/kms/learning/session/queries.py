"""Owner-scoped Cypher for learning sessions, tasks, and assessments."""

CREATE_SESSION = """
MATCH (user:User {uuid: $user_uuid})
CREATE (session:LearningSession {
    uuid: $uuid, goal: $goal, targets: $targets, status: $status,
    created_at: $created_at, updated_at: $updated_at,
    invalidated_at: $invalidated_at
})
CREATE (user)-[:OWNS_SESSION {
    created_at: $created_at, updated_at: $created_at
}]->(session)
RETURN session.uuid AS uuid
"""

READ_SESSION = """
MATCH (:User {uuid: $user_uuid})-[:OWNS_SESSION]->(session:LearningSession {uuid: $session_uuid})
CALL (session) {
    OPTIONAL MATCH (session)-[:FIRST_STEP]->(first:LearningSessionStep)
    OPTIONAL MATCH step_path = (first)-[:NEXT_STEP*0..]->(step:LearningSessionStep)
    WHERE step IS NULL OR (
        (session)-[:HAS_STEP]->(step)
        AND all(path_step IN nodes(step_path) WHERE (session)-[:HAS_STEP]->(path_step))
    )
    WITH step, step_path ORDER BY length(step_path)
    CALL (step) {
        OPTIONAL MATCH (step)-[:GROUNDED_IN]->(proposition:Proposition)
        RETURN collect(proposition.uuid) AS proposition_uuids
    }
    CALL (step) {
        OPTIONAL MATCH (step)-[:GROUNDED_IN]->(argument:Argument)
        RETURN collect(argument.uuid) AS argument_uuids
    }
    CALL (step) {
        OPTIONAL MATCH (step)-[:GROUNDED_IN]->(unit:SourceUnit)
        RETURN collect(unit.uuid) AS unit_uuids
    }
    CALL (step) {
        OPTIONAL MATCH (step)-[:ASSESSES|TEACHES|TARGETS]->(target)
        WHERE target:Proposition OR target:Argument OR target:Solution
        RETURN head(collect(CASE WHEN target IS NULL THEN NULL ELSE {
            kind: CASE
                WHEN target:Proposition THEN 'proposition'
                WHEN target:Argument THEN 'argument'
                ELSE 'solution'
            END,
            uuid: target.uuid
        } END)) AS target
    }
    CALL (step) {
        OPTIONAL MATCH (step)-[:HAS_SOLUTION]->(solution:Solution)
        WITH solution
        CALL (solution) {
            OPTIONAL MATCH (solution)-[:GROUNDED_IN]->(proposition:Proposition)
            RETURN collect(proposition.uuid) AS proposition_uuids
        }
        CALL (solution) {
            OPTIONAL MATCH (solution)-[:GROUNDED_IN]->(argument:Argument)
            RETURN collect(argument.uuid) AS argument_uuids
        }
        CALL (solution) {
            OPTIONAL MATCH (solution)-[:GROUNDED_IN]->(unit:SourceUnit)
            RETURN collect(unit.uuid) AS unit_uuids
        }
        RETURN head(collect(CASE WHEN solution IS NULL THEN NULL ELSE {
            uuid: solution.uuid, payload_json: solution.payload_json,
            created_at: solution.created_at, updated_at: solution.updated_at,
            invalidated_at: solution.invalidated_at,
            proposition_uuids: proposition_uuids,
            argument_uuids: argument_uuids, unit_uuids: unit_uuids
        } END)) AS raw_solution
    }
    CALL (step) {
        OPTIONAL MATCH (step)-[:BASED_ON]->(assessment:TaskAssessment)
        RETURN collect(assessment.uuid) AS assessment_uuids
    }
    CALL (step) {
        OPTIONAL MATCH (step)-[:HAS_RESPONSE]->(response:TaskResponse)
        WITH step, response ORDER BY response.created_at, response.uuid
        CALL (response) {
            OPTIONAL MATCH (response)-[:HAS_ASSESSMENT]->(assessment:TaskAssessment)
            WITH assessment ORDER BY assessment.created_at, assessment.uuid
            RETURN collect(CASE WHEN assessment IS NULL THEN NULL ELSE {
                uuid: assessment.uuid, created_at: assessment.created_at,
                updated_at: assessment.updated_at, invalidated_at: assessment.invalidated_at,
                response_uuid: response.uuid, payload_json: assessment.payload_json
            } END) AS raw_assessments
        }
        RETURN collect(CASE WHEN response IS NULL THEN NULL ELSE {
            uuid: response.uuid, created_at: response.created_at,
            updated_at: response.updated_at, invalidated_at: response.invalidated_at,
            task_uuid: step.uuid, payload_json: response.payload_json,
            assessments: [item IN raw_assessments WHERE item IS NOT NULL]
        } END) AS raw_responses
    }
    RETURN collect(CASE WHEN step IS NULL THEN NULL ELSE {
        uuid: step.uuid,
        kind: CASE
            WHEN step:Task THEN 'task'
            WHEN step:Teaching THEN 'teaching'
            ELSE 'decision'
        END,
        created_at: step.created_at, updated_at: step.updated_at,
        invalidated_at: step.invalidated_at, payload_json: step.payload_json,
        target: target,
        proposition_uuids: proposition_uuids,
        argument_uuids: argument_uuids, unit_uuids: unit_uuids,
        solution: raw_solution, assessment_uuids: assessment_uuids,
        responses: [item IN raw_responses WHERE item IS NOT NULL]
    } END) AS raw_steps
}
RETURN session.uuid AS uuid, session.goal AS goal, session.targets AS targets,
       session.status AS status, session.created_at AS created_at,
       session.updated_at AS updated_at, session.invalidated_at AS invalidated_at,
       [item IN raw_steps WHERE item IS NOT NULL] AS steps
"""

APPEND_TEACHING_ASSOCIATIONS = """
MATCH (user:User {uuid: $user_uuid})-[:OWNS_SESSION]->(session:LearningSession {uuid: $session_uuid})
MATCH (session)-[:HAS_STEP]->(teaching:Teaching {uuid: $teaching_uuid})
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
CREATE (teaching)-[:TEACHES {created_at: $created_at, updated_at: $created_at}]->(target)
WITH user, session, teaching
OPTIONAL MATCH (user)-[:OWNS_PROPOSITION]->(proposition:Proposition)
WHERE proposition.uuid IN $proposition_uuids
WITH user, session, teaching, collect(DISTINCT proposition) AS propositions
FOREACH (node IN propositions |
    CREATE (teaching)-[:GROUNDED_IN {created_at: $created_at, updated_at: $created_at}]->(node)
)
WITH user, session, teaching, propositions
WHERE size(propositions) = size($proposition_uuids)
OPTIONAL MATCH (user)-[:OWNS_ARGUMENT]->(argument:Argument)
WHERE argument.uuid IN $argument_uuids
WITH user, session, teaching, propositions, collect(DISTINCT argument) AS arguments
FOREACH (node IN arguments |
    CREATE (teaching)-[:GROUNDED_IN {created_at: $created_at, updated_at: $created_at}]->(node)
)
WITH user, session, teaching, propositions, arguments
WHERE size(arguments) = size($argument_uuids)
OPTIONAL MATCH (user)-[:OWNS_SOURCE]->(:Source)-[:HAS_UNIT]->(unit:SourceUnit)
WHERE unit.uuid IN $unit_uuids
WITH session, teaching, propositions, arguments, collect(DISTINCT unit) AS units
FOREACH (node IN units |
    CREATE (teaching)-[:GROUNDED_IN {created_at: $created_at, updated_at: $created_at}]->(node)
)
WITH session, teaching
WHERE size(units) = size($unit_uuids)
CALL (session, teaching) {
    UNWIND $assessment_uuids AS ref
    MATCH (session)-[:HAS_STEP]->(:Task)-[:HAS_RESPONSE]->(:TaskResponse)-[:HAS_ASSESSMENT]->(assessment:TaskAssessment {uuid: ref})
    CREATE (teaching)-[:BASED_ON {created_at: $created_at, updated_at: $created_at}]->(assessment)
    RETURN count(DISTINCT assessment) AS matched
}
WITH teaching, matched WHERE matched = size($assessment_uuids)
RETURN teaching.uuid AS uuid
"""

CREATE_TASK_SOLUTION = """
MATCH (user:User {uuid: $user_uuid})-[:OWNS_SESSION]->(session:LearningSession {uuid: $session_uuid})
MATCH (session)-[:HAS_STEP]->(task:Task {uuid: $task_uuid})
CREATE (solution:Solution {
    uuid: $solution_uuid, payload_json: $solution_payload_json,
    created_at: $solution_created_at, updated_at: $solution_updated_at,
    invalidated_at: $solution_invalidated_at
})
CREATE (task)-[:HAS_SOLUTION {created_at: $created_at, updated_at: $created_at}]->(solution)
WITH user, task, solution
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
CREATE (task)-[:ASSESSES {created_at: $created_at, updated_at: $created_at}]->(target)
WITH user, task, solution
OPTIONAL MATCH (user)-[:OWNS_PROPOSITION]->(task_proposition:Proposition)
WHERE task_proposition.uuid IN $task_proposition_uuids
WITH user, task, solution, collect(DISTINCT task_proposition) AS task_propositions
FOREACH (node IN task_propositions |
    CREATE (task)-[:GROUNDED_IN {created_at: $created_at, updated_at: $created_at}]->(node)
)
WITH user, task, solution, task_propositions
WHERE size(task_propositions) = size($task_proposition_uuids)
OPTIONAL MATCH (user)-[:OWNS_ARGUMENT]->(task_argument:Argument)
WHERE task_argument.uuid IN $task_argument_uuids
WITH user, task, solution, task_propositions, collect(DISTINCT task_argument) AS task_arguments
FOREACH (node IN task_arguments |
    CREATE (task)-[:GROUNDED_IN {created_at: $created_at, updated_at: $created_at}]->(node)
)
WITH user, task, solution, task_propositions, task_arguments
WHERE size(task_arguments) = size($task_argument_uuids)
OPTIONAL MATCH (user)-[:OWNS_SOURCE]->(:Source)-[:HAS_UNIT]->(task_unit:SourceUnit)
WHERE task_unit.uuid IN $task_unit_uuids
WITH user, task, solution, task_propositions, task_arguments, collect(DISTINCT task_unit) AS task_units
FOREACH (node IN task_units |
    CREATE (task)-[:GROUNDED_IN {created_at: $created_at, updated_at: $created_at}]->(node)
)
WITH user, task, solution, task_propositions, task_arguments, task_units
WHERE size(task_units) = size($task_unit_uuids)
OPTIONAL MATCH (user)-[:OWNS_PROPOSITION]->(solution_proposition:Proposition)
WHERE solution_proposition.uuid IN $solution_proposition_uuids
WITH user, task, solution, task_propositions, task_arguments, task_units, collect(DISTINCT solution_proposition) AS solution_propositions
FOREACH (node IN solution_propositions |
    CREATE (solution)-[:GROUNDED_IN {created_at: $created_at, updated_at: $created_at}]->(node)
)
WITH user, task, solution, task_propositions, task_arguments, task_units, solution_propositions
WHERE size(solution_propositions) = size($solution_proposition_uuids)
OPTIONAL MATCH (user)-[:OWNS_ARGUMENT]->(solution_argument:Argument)
WHERE solution_argument.uuid IN $solution_argument_uuids
WITH user, task, solution, task_propositions, task_arguments, task_units, solution_propositions, collect(DISTINCT solution_argument) AS solution_arguments
FOREACH (node IN solution_arguments |
    CREATE (solution)-[:GROUNDED_IN {created_at: $created_at, updated_at: $created_at}]->(node)
)
WITH user, task, solution, task_propositions, task_arguments, task_units, solution_propositions, solution_arguments
WHERE size(solution_arguments) = size($solution_argument_uuids)
OPTIONAL MATCH (user)-[:OWNS_SOURCE]->(:Source)-[:HAS_UNIT]->(solution_unit:SourceUnit)
WHERE solution_unit.uuid IN $solution_unit_uuids
WITH task, solution, task_propositions, task_arguments, task_units, solution_propositions, solution_arguments, collect(DISTINCT solution_unit) AS solution_units
FOREACH (node IN solution_units |
    CREATE (solution)-[:GROUNDED_IN {created_at: $created_at, updated_at: $created_at}]->(node)
)
WITH task, solution, solution_units
WHERE size(solution_units) = size($solution_unit_uuids)
RETURN task.uuid AS task_uuid, solution.uuid AS solution_uuid
"""

READ_OWNED_TASK = """
MATCH (:User {uuid: $user_uuid})-[:OWNS_SESSION]->(session:LearningSession {uuid: $session_uuid})
MATCH (session)-[:HAS_STEP]->(task:Task {uuid: $task_uuid})
OPTIONAL MATCH (task)-[:ASSESSES]->(target)
WHERE target:Proposition OR target:Argument OR target:Solution
WITH session, task, head(collect(CASE WHEN target IS NULL THEN NULL ELSE {
    kind: CASE
        WHEN target:Proposition THEN 'proposition'
        WHEN target:Argument THEN 'argument'
        ELSE 'solution'
    END,
    uuid: target.uuid
} END)) AS target
CALL (task) {
    OPTIONAL MATCH (task)-[:GROUNDED_IN]->(proposition:Proposition)
    RETURN collect(proposition.uuid) AS proposition_uuids
}
CALL (task) {
    OPTIONAL MATCH (task)-[:GROUNDED_IN]->(argument:Argument)
    RETURN collect(argument.uuid) AS argument_uuids
}
CALL (task) {
    OPTIONAL MATCH (task)-[:GROUNDED_IN]->(unit:SourceUnit)
    RETURN collect(unit.uuid) AS unit_uuids
}
CALL (task) {
    MATCH (task)-[:HAS_SOLUTION]->(solution:Solution)
    CALL (solution) {
        OPTIONAL MATCH (solution)-[:GROUNDED_IN]->(proposition:Proposition)
        RETURN collect(proposition.uuid) AS proposition_uuids
    }
    CALL (solution) {
        OPTIONAL MATCH (solution)-[:GROUNDED_IN]->(argument:Argument)
        RETURN collect(argument.uuid) AS argument_uuids
    }
    CALL (solution) {
        OPTIONAL MATCH (solution)-[:GROUNDED_IN]->(unit:SourceUnit)
        RETURN collect(unit.uuid) AS unit_uuids
    }
    RETURN collect({
        uuid: solution.uuid, payload_json: solution.payload_json,
        created_at: solution.created_at, updated_at: solution.updated_at,
        invalidated_at: solution.invalidated_at,
        proposition_uuids: proposition_uuids,
        argument_uuids: argument_uuids, unit_uuids: unit_uuids
    })[0] AS solution
}
RETURN task.uuid AS uuid, task.payload_json AS payload_json,
       task.created_at AS created_at, task.updated_at AS updated_at,
       task.invalidated_at AS invalidated_at, target,
       proposition_uuids, argument_uuids, unit_uuids, solution
"""

LOCK_SESSION_FOR_APPEND = """
MATCH (:User {uuid: $user_uuid})-[:OWNS_SESSION]->(session:LearningSession {uuid: $session_uuid})
SET session.updated_at = session.updated_at
RETURN session.uuid AS uuid, session.status AS status
"""

APPEND_STEP = """
MATCH (:User {uuid: $user_uuid})-[:OWNS_SESSION]->(session:LearningSession {uuid: $session_uuid})
WITH session WHERE session.status = 'active'
OPTIONAL MATCH (session)-[:LAST_STEP]->(last:LearningSessionStep)
CREATE (step:LearningSessionStep:$step_label {
    uuid: $uuid, payload_json: $payload_json,
    created_at: $created_at, updated_at: $updated_at,
    invalidated_at: $invalidated_at
})
CREATE (session)-[:HAS_STEP {created_at: $created_at, updated_at: $created_at}]->(step)
WITH session, last, step
OPTIONAL MATCH (session)-[old:LAST_STEP]->(last)
DELETE old
WITH session, last, step
FOREACH (_ IN CASE WHEN last IS NULL THEN [1] ELSE [] END |
    CREATE (session)-[:FIRST_STEP {created_at: $created_at, updated_at: $created_at}]->(step)
)
FOREACH (_ IN CASE WHEN last IS NOT NULL THEN [1] ELSE [] END |
    CREATE (last)-[:NEXT_STEP {created_at: $created_at, updated_at: $created_at}]->(step)
)
CREATE (session)-[:LAST_STEP {created_at: $created_at, updated_at: $created_at}]->(step)
SET session.updated_at = $created_at,
    session.status = CASE WHEN $is_finish THEN 'completed' ELSE session.status END
RETURN step.uuid AS uuid
"""


ATTACH_DECISION_ASSOCIATIONS = """
MATCH (:User {uuid: $user_uuid})-[:OWNS_SESSION]->(session:LearningSession {uuid: $session_uuid})
MATCH (session)-[:HAS_STEP]->(decision:LearningDecision {uuid: $decision_uuid})
CALL (session, decision) {
    UNWIND $assessment_uuids AS ref
    MATCH (session)-[:HAS_STEP]->(:Task)-[:HAS_RESPONSE]->(:TaskResponse)-[:HAS_ASSESSMENT]->(assessment:TaskAssessment {uuid: ref})
    CREATE (decision)-[:BASED_ON {created_at: $created_at, updated_at: $created_at}]->(assessment)
    RETURN count(DISTINCT assessment) AS matched
}
WITH session, decision, matched
WHERE matched = size($assessment_uuids)
CALL (session, decision) {
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
        UNION
        WITH user WHERE $target_kind IS NULL
        RETURN null AS target
    }
    FOREACH (_ IN CASE WHEN target IS NULL THEN [] ELSE [1] END |
        CREATE (decision)-[:TARGETS {created_at: $created_at, updated_at: $created_at}]->(target)
    )
    RETURN count(target) AS targets
}
WITH decision, targets
WHERE targets = CASE WHEN $target_kind IS NULL THEN 0 ELSE 1 END
RETURN decision.uuid AS uuid
"""

ATTACH_TASK_RESPONSE = """
MATCH (:User {uuid: $user_uuid})-[:OWNS_SESSION]->(session:LearningSession {uuid: $session_uuid})
SET session.updated_at = session.updated_at
WITH session WHERE session.status = 'active'
MATCH (session)-[:HAS_STEP]->(task:Task {uuid: $task_uuid})
CREATE (response:TaskResponse {
    uuid: $uuid, payload_json: $payload_json,
    created_at: $created_at, updated_at: $updated_at,
    invalidated_at: $invalidated_at
})
CREATE (task)-[:HAS_RESPONSE {created_at: $created_at, updated_at: $created_at}]->(response)
RETURN response.uuid AS uuid
"""

READ_OWNED_TASK_RESPONSE = """
MATCH (:User {uuid: $user_uuid})-[:OWNS_SESSION]->(session:LearningSession {uuid: $session_uuid})
MATCH (session)-[:HAS_STEP]->(task:Task)-[:HAS_RESPONSE]->(response:TaskResponse {uuid: $response_uuid})
RETURN response.uuid AS uuid, response.payload_json AS payload_json,
       response.created_at AS created_at, response.updated_at AS updated_at,
       response.invalidated_at AS invalidated_at, task.uuid AS task_uuid
"""

ATTACH_TASK_ASSESSMENT = """
MATCH (:User {uuid: $user_uuid})-[:OWNS_SESSION]->(session:LearningSession {uuid: $session_uuid})
SET session.updated_at = session.updated_at
WITH session WHERE session.status = 'active'
MATCH (session)-[:HAS_STEP]->(:Task)-[:HAS_RESPONSE]->(response:TaskResponse {uuid: $response_uuid})
CREATE (assessment:TaskAssessment {
    uuid: $uuid, payload_json: $payload_json,
    created_at: $created_at, updated_at: $updated_at,
    invalidated_at: $invalidated_at
})
CREATE (response)-[:HAS_ASSESSMENT {created_at: $created_at, updated_at: $created_at}]->(assessment)
RETURN assessment.uuid AS uuid
"""
