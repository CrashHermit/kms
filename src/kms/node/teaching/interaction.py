"""Initialize the evidence workspace for the flat teaching workflow."""

from kms.core.source_workspace import append_unique
from kms.langgraph.teaching.state import TeachingTurnState
from kms.node.teaching.evidence import register_learner_observation
from kms.service.learner_observations import LearnerObservationService
from kms.service.source_evidence import SourceEvidenceService
from kms.service.teaching_workspace import restore_workspace


class TeachingInteractionNode:
    """Restore persisted evidence and activate retrieved learner history."""

    def __init__(
        self,
        source_evidence: SourceEvidenceService,
        learner_observations: LearnerObservationService,
    ) -> None:
        self._source_evidence = source_evidence
        self._learner_observations = learner_observations

    async def run(self, state: TeachingTurnState) -> dict[str, object]:
        """Restore the workspace that remains live through teaching phases."""
        workspace = await restore_workspace(
            state.request.snapshot.workspace,
            user_uuid=state.request.user_uuid,
            source_evidence=self._source_evidence,
            learner_observations=self._learner_observations,
        )
        automatic_positions = []
        for record in state.learner_history:
            position = register_learner_observation(
                workspace, record.observation
            )
            workspace.learner_observation_registry[position].record = record
            append_unique(automatic_positions, position)
        active_positions = automatic_positions.copy()
        for position in workspace.active_learner_observation_positions:
            append_unique(active_positions, position)
        workspace.active_learner_observation_positions = active_positions
        return {'workspace': workspace}
