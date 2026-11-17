"""Select explicitly requested session history for inference."""

from kms.learning.session.models import SessionContext, SessionHistory


def select_session_context(
    history: SessionHistory, step_uuids: list[str]
) -> SessionContext:
    """Keep selected steps and their associated learner records in stored order."""
    selected = set(step_uuids)
    steps = [step for step in history.steps if step.uuid in selected]
    task_ids = {step.uuid for step in steps if step.kind == 'task'}
    responses = [
        response
        for response in history.responses
        if response.task_uuid in task_ids
    ]
    response_ids = {response.uuid for response in responses}
    assessments = [
        assessment
        for assessment in history.assessments
        if assessment.response_uuid in response_ids
    ]
    return SessionContext(
        goal=history.session.goal,
        targets=history.session.targets,
        steps=steps,
        responses=responses,
        assessments=assessments,
    )
