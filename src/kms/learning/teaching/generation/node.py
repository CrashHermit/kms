"""Generate and durably append an owner-scoped teaching step."""

from kms.learning.materials import resolve_material
from kms.learning.session.repository import LearningSessionRepository
from kms.learning.teaching.generation.inference import TeachingGenerationModule
from kms.learning.teaching.generation.models import (
    TeachingGenerationRequest,
    TeachingState,
)
from kms.learning.teaching.models import Teaching
from kms.media.content import collect_image_assets


class TeachingGenerationNode:
    """Return generated teaching only after its session append succeeds."""

    def __init__(
        self,
        module: TeachingGenerationModule,
        repository: LearningSessionRepository,
    ) -> None:
        self._module = module
        self._repository = repository

    async def run(self, state: TeachingState) -> dict[str, Teaching]:
        await self._repository.read_session(
            state['user_uuid'], state['session_uuid']
        )
        request: TeachingGenerationRequest = state['request']
        draft = await self._module.acall(request=request)
        teaching = Teaching(
            understanding=request.understanding,
            objective=request.objective,
            target=request.target,
            material=resolve_material(
                draft.material, collect_image_assets({'request': request})
            ),
            grounding=draft.grounding,
            assessment_uuids=request.assessment_uuids,
        )
        persisted = await self._repository.append_teaching(
            state['user_uuid'], state['session_uuid'], teaching
        )
        return {'teaching': persisted}
