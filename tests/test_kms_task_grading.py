"""Owner-scoped grading of persisted free-response task attempts."""

import asyncio

import pytest

from kms.learning.references import GroundingReferences, SolutionTarget
from kms.learning.task.grading.models import GradingState, TaskGradingRequest
from kms.learning.task.grading.node import TaskGradingNode
from kms.learning.task.models import (
    Confidence,
    FreeAnswer,
    ResponseInterpretation,
    ReviewRating,
    Solution,
    Task,
    TaskAssessment,
    TaskGrade,
    TaskResponse,
)
from kms.media.models import Material


def _task(uuid: str = 'actual-task') -> Task:
    return Task(
        uuid=uuid,
        understanding='Differentiate a square.',
        prompt=Material.from_text('Differentiate x².'),
        target=SolutionTarget(uuid='reference-solution'),
        grounding=GroundingReferences(),
        solution=Solution(
            uuid='reference-solution',
            material=Material.from_text('2x'),
            assessment_criteria=['Equivalent derivative.'],
            grounding=GroundingReferences(),
        ),
    )


def _response(task_uuid: str = 'actual-task') -> TaskResponse:
    return TaskResponse(
        uuid='stored-response',
        task_uuid=task_uuid,
        answer=FreeAnswer(
            material=Material.from_text(
                '2x, because differentiating always doubles the variable.'
            ),
            confidence=Confidence.HIGH,
        ),
    )


def _grade() -> TaskGrade:
    return TaskGrade(
        correctness='correct',
        interpretation=ResponseInterpretation(
            reasoning_verdict='unsound',
            explanation='The answer is correct but the stated reasoning is flawed.',
            demonstrated_understandings=[],
            gaps=[],
            possible_misconceptions=[],
            unresolved_questions=[],
        ),
        review_rating=ReviewRating.AGAIN,
    )


class _Repository:
    def __init__(self, response: TaskResponse, task: Task) -> None:
        self.response = response
        self.task = task
        self.read_task_uuid: str | None = None
        self.persisted: TaskAssessment | None = None
        self.fail_read = False
        self.fail_write = False

    async def read_task_response(
        self, user_uuid: str, session_uuid: str, response_uuid: str
    ) -> TaskResponse:
        assert (user_uuid, session_uuid, response_uuid) == (
            'owner',
            'session',
            'response-id',
        )
        if self.fail_read:
            raise LookupError('foreign response')
        return self.response

    async def read_task(
        self, user_uuid: str, session_uuid: str, task_uuid: str
    ) -> Task:
        assert (user_uuid, session_uuid) == ('owner', 'session')
        assert task_uuid == self.task.uuid
        self.read_task_uuid = task_uuid
        return self.task

    async def record_task_assessment(
        self, user_uuid: str, session_uuid: str, assessment: TaskAssessment
    ) -> TaskAssessment:
        assert (user_uuid, session_uuid) == ('owner', 'session')
        if self.fail_write:
            raise RuntimeError('write failed')
        self.persisted = assessment.model_copy(update={'uuid': 'persisted'})
        return self.persisted


class _Module:
    def __init__(self, grade: TaskGrade) -> None:
        self.grade = grade
        self.request: TaskGradingRequest | None = None
        self.called = False

    async def acall(self, *, request: TaskGradingRequest) -> TaskGrade:
        self.called = True
        self.request = request
        return self.grade


def _state() -> GradingState:
    return {
        'user_uuid': 'owner',
        'session_uuid': 'session',
        'response_uuid': 'response-id',
    }


def test_grading_uses_stored_response_task_and_returns_persisted_assessment() -> (
    None
):
    response = _response()
    task = _task('actual-task')
    repository = _Repository(response, task)
    module = _Module(_grade())
    node = TaskGradingNode(module, repository)  # type: ignore[arg-type]

    result = asyncio.run(node.run(_state()))['assessment']

    assert repository.read_task_uuid == response.task_uuid
    assert module.request is not None
    assert module.request.response is response
    assert module.request.task is task
    assert repository.persisted is result
    assert result.response_uuid == response.uuid
    assert result.grade == module.grade
    assert result.uuid == 'persisted'


def test_foreign_response_fails_before_inference() -> None:
    repository = _Repository(_response(), _task())
    repository.fail_read = True
    module = _Module(_grade())
    node = TaskGradingNode(module, repository)  # type: ignore[arg-type]

    with pytest.raises(LookupError, match='foreign response'):
        asyncio.run(node.run(_state()))

    assert not module.called


def test_assessment_write_failure_propagates() -> None:
    repository = _Repository(_response(), _task())
    repository.fail_write = True
    module = _Module(_grade())
    node = TaskGradingNode(module, repository)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match='write failed'):
        asyncio.run(node.run(_state()))

    assert module.called
    assert repository.persisted is None
