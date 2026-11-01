"""Adapt one evidence action to the bounded teaching interaction state."""

from typing import cast

from kms2.core.model.teaching.inference import EvidenceToolAction
from kms2.core.model.teaching.operation import RespondAction
from kms2.core.model.teaching.workflow import (
    EvidenceExecutionRequest,
    TeachingInteractionResult,
)
from kms2.langgraph.evidence_acquisition.graph import EvidenceAcquisitionGraph
from kms2.langgraph.teaching.interaction_state import TeachingInteractionState


class TeachingInteractionToolsNode:
    """Execute one action and own interaction-local observation bookkeeping."""

    def __init__(self, evidence: EvidenceAcquisitionGraph) -> None:
        self._evidence = evidence

    async def run(self, state: TeachingInteractionState) -> dict[str, object]:
        """Run the latest non-response action once and append its observation."""
        action = state.latest_action
        assert action is not None and not isinstance(action, RespondAction)
        result = await self._evidence.run(
            EvidenceExecutionRequest(
                user_uuid=state.request.turn.user_uuid,
                source_uuids=state.request.turn.source_uuids,
                action=cast(EvidenceToolAction, action),
                workspace=state.workspace,
            )
        )
        workspace = result.workspace
        workspace.observations.append(result.observation)
        return {'workspace': workspace, 'tool_calls': state.tool_calls + 1}

    @staticmethod
    def finish(state: TeachingInteractionState) -> dict[str, object]:
        """Project the completed response and its final visible-position maps."""
        response = state.latest_action
        assert isinstance(response, RespondAction)
        return {
            'result': TeachingInteractionResult(
                response=response,
                workspace=state.workspace,
                visible_turn_indexes=state.visible_turn_indexes,
                visible_observation_indexes=state.visible_observation_indexes,
                visible_citation_receipt_indexes=(
                    state.visible_citation_receipt_indexes
                ),
            )
        }
