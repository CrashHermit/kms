"""Prepare learner-observation writes from one completed response."""

from kms.core.embedding import Embedder
from kms.core.model.content import MultimodalContent
from kms.core.model.teaching.observation import (
    LearnerObservationWrite,
    observation_text,
)
from kms.langgraph.teaching.state import TeachingTurnState
from kms.service.learner_observations import LearnerObservationService


class TeachingObservationNode:
    """Embed response observations and resolve their visible evidence turns."""

    def __init__(
        self,
        observations: LearnerObservationService,
        embedder: Embedder,
    ) -> None:
        self._observations = observations
        self._embedder = embedder

    async def run(self, state: TeachingTurnState) -> dict[str, object]:
        """Prepare durable observation records without performing persistence."""
        interaction = state.interaction
        assert interaction is not None
        drafts = interaction.response.learner_observations
        if drafts:
            embeddings = await self._embedder.embed(
                [
                    MultimodalContent(
                        text=observation_text(
                            draft.kind, draft.topic, draft.description
                        )
                    )
                    for draft in drafts
                ],
                content_images=[],
                input_type='document',
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
