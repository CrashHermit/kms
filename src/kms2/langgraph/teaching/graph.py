"""Acyclic parent graph for one complete teaching turn."""

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from kms2.core.model.teaching.session import TeachingSessionView
from kms2.core.model.teaching.workflow import TeachingTurnRequest
from kms2.langgraph.teaching.state import TeachingTurnState
from kms2.langgraph.teaching.turn import add_teaching_turn_phase
from kms2.node.teaching.history import TeachingHistoryNode
from kms2.node.teaching.interaction import TeachingInteractionNode
from kms2.node.teaching.knowledge import TeachingKnowledgeNode
from kms2.node.teaching.observation import TeachingObservationNode
from kms2.node.teaching.persistence import TeachingPersistenceNode


class TeachingGraph:
    """Compile and invoke the fixed parent workflow for one teacher turn."""

    def __init__(
        self,
        history: TeachingHistoryNode,
        interaction: TeachingInteractionNode,
        observations: TeachingObservationNode,
        knowledge: TeachingKnowledgeNode,
        persistence: TeachingPersistenceNode,
    ) -> None:
        self._history = history
        self._interaction = interaction
        self._observations = observations
        self._knowledge = knowledge
        self._persistence = persistence
        self._compiled_graph: CompiledStateGraph | None = None

    def build_graph(self) -> CompiledStateGraph:
        """Build and cache the five-node parent graph."""
        if self._compiled_graph is None:
            graph = StateGraph(TeachingTurnState)
            add_teaching_turn_phase(
                graph,
                self._history,
                self._interaction,
                self._observations,
                self._knowledge,
            )
            graph.add_node('teaching_persistence', self._persistence.run)
            graph.add_edge('teaching_knowledge', 'teaching_persistence')
            graph.add_edge('teaching_persistence', END)
            self._compiled_graph = graph.compile()
        return self._compiled_graph

    async def run(self, request: TeachingTurnRequest) -> TeachingSessionView:
        """Run one retrieval, interaction, preparation, and commit sequence."""
        result = await self.build_graph().ainvoke(
            {'request': request},
            config={'recursion_limit': 10},
        )
        final_view = TeachingTurnState.model_validate(result).final_view
        assert final_view is not None
        return final_view
