"""Run the bounded instructional interaction as one parent turn stage."""

from kms2.core.model.teaching.workflow import (
    TeachingInteractionRequest,
    TeachingInteractionResult,
)
from kms2.langgraph.teaching.interaction_graph import TeachingInteractionGraph
from kms2.langgraph.teaching.state import TeachingTurnState


class TeachingInteractionNode:
    """Invoke the private decision loop with retrieved learner history."""

    def __init__(self, graph: TeachingInteractionGraph) -> None:
        self._graph = graph

    async def run(self, state: TeachingTurnState) -> dict[str, object]:
        """Run one interaction and return its final typed result."""
        request = TeachingInteractionRequest(
            turn=state.request,
            learner_history=state.learner_history,
        )
        interaction: TeachingInteractionResult = await self._graph.run(request)
        return {'interaction': interaction}
