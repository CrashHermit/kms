"""Prepare learner-observation writes from one completed response."""

from kms2.core.embedding import EmbeddingClient
from kms2.core.model.teaching.observation import (
    LearnerObservationWrite,
    observation_text,
)
from kms2.langgraph.teaching.state import TeachingTurnState
from kms2.service.learner_observations import LearnerObservationService


class TeachingObservationNode:
    """Embed response observations and resolve their visible evidence turns."""

    def __init__(
        self,
        observations: LearnerObservationService,
        embedding_client: EmbeddingClient,
    ) -> None:
        self._observations = observations
        self._embedding_client = embedding_client

    async def run(self, state: TeachingTurnState) -> dict[str, object]:
        """Prepare durable observation records without performing persistence."""
        interaction = state.interaction
        assert interaction is not None
        drafts = interaction.response.learner_observations
        if drafts:
            embeddings = await self._embedding_client.embed(
                [
                    observation_text(draft.kind, draft.topic, draft.description)
                    for draft in drafts
                ]
            )
        else:
            embeddings = []

        prepared: list[LearnerObservationWrite] = self._observations.prepare(
            drafts,
            state.request.snapshot,
            interaction.visible_turn_indexes,
            embeddings,
        )
        return {'prepared_observations': prepared}
