"""Grade a stored response against its actual owner-scoped task."""

from kms.learning.session.repository import LearningSessionRepository
from kms.learning.task.grading.inference import TaskGradingModule
from kms.learning.task.grading.models import GradingState, TaskGradingRequest
from kms.learning.task.models import TaskAssessment


class TaskGradingNode:
    """Persist an assessment before returning it to the caller."""

    def __init__(
        self,
        module: TaskGradingModule,
        repository: LearningSessionRepository,
    ) -> None:
        self._module = module
        self._repository = repository

    async def run(self, state: GradingState) -> dict[str, TaskAssessment]:
        response = await self._repository.read_task_response(
            state['user_uuid'], state['session_uuid'], state['response_uuid']
        )
        task = await self._repository.read_task(
            state['user_uuid'], state['session_uuid'], response.task_uuid
        )
        grade = await self._module.acall(
            request=TaskGradingRequest(task=task, response=response)
        )
        assessment = TaskAssessment(response_uuid=response.uuid, grade=grade)
        persisted = await self._repository.record_task_assessment(
            state['user_uuid'], state['session_uuid'], assessment
        )
        return {'assessment': persisted}
