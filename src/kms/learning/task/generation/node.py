"""Generate and persist an owner-scoped free-response task."""

from kms.learning.materials import resolve_material
from kms.learning.references import SolutionTarget
from kms.learning.session.repository import LearningSessionRepository
from kms.learning.task.generation.inference import TaskGenerationModule
from kms.learning.task.generation.models import (
    NewSolutionTarget,
    TaskDraft,
    TaskState,
)
from kms.learning.task.models import Solution, Task
from kms.media.content import collect_image_assets


def _task(draft: TaskDraft, state: TaskState) -> Task:
    """Build a task whose assessed identity comes from its request."""
    request = state['request']
    image_assets = collect_image_assets({'request': request})
    solution = Solution(
        material=resolve_material(draft.solution.material, image_assets),
        assessment_criteria=draft.solution.assessment_criteria,
        grounding=draft.solution.grounding,
    )
    target = (
        SolutionTarget(uuid=solution.uuid)
        if isinstance(request.target, NewSolutionTarget)
        else request.target
    )
    return Task(
        understanding=request.understanding,
        prompt=resolve_material(draft.prompt, image_assets),
        target=target,
        grounding=draft.grounding,
        solution=solution,
    )


class TaskGenerationNode:
    """Return a task only after its owner-scoped persistence succeeds."""

    def __init__(
        self,
        module: TaskGenerationModule,
        repository: LearningSessionRepository,
    ) -> None:
        self._module = module
        self._repository = repository

    async def run(self, state: TaskState) -> dict[str, Task]:
        await self._repository.read_session(
            state['user_uuid'], state['session_uuid']
        )
        draft = await self._module.acall(request=state['request'])
        task = _task(draft, state)
        persisted_task = await self._repository.append_task(
            state['user_uuid'], state['session_uuid'], task
        )
        return {'task': persisted_task}
