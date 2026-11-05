"""Retrieve bounded learner history for one teaching turn."""

from kms.core.model.teaching.observation import LearnerObservationRecord
from kms.core.model.teaching.workflow import TeachingTurnRequest
from kms.langgraph.teaching.state import TeachingTurnState
from kms.service.learner_observation_search import (
    LearnerObservationSearchService,
)


class TeachingHistoryNode:
    """Load ranked learner observations relevant to the current goal and step."""

    def __init__(self, search: LearnerObservationSearchService) -> None:
        self._search = search

    async def run(self, state: TeachingTurnState) -> dict[str, object]:
        """Retrieve history once before the interaction loop begins."""
        request: TeachingTurnRequest = state.request
        snapshot = request.snapshot
        query_parts = [snapshot.session.goal]
        if snapshot.current_plan is not None and snapshot.current_step_uuid:
            current_step = next(
                (
                    step
                    for step in snapshot.current_plan.steps
                    if step.uuid == snapshot.current_step_uuid
                ),
                None,
            )
            if current_step is not None:
                query_parts.extend((current_step.title, current_step.objective))

        hits = await self._search.search(
            request.user_uuid,
            '\n'.join(part for part in query_parts if part),
        )
        records: list[LearnerObservationRecord] = [hit.record for hit in hits]
        return {'learner_history': records}
