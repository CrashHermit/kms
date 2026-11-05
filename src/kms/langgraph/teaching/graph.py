"""Flat LangGraph workflow for one complete teaching turn."""

from collections.abc import Callable

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from kms.core.model.research.inference import FinishResearchAction
from kms.core.model.teaching.session import TeachingPhase
from kms.langgraph.teaching.interaction.decision import (
    add_teaching_decision_phase,
)
from kms.langgraph.teaching.state import TeachingTurnState
from kms.node.research.researcher import ResearcherNode
from kms.node.research.tools import ResearchToolsNode
from kms.node.teaching.history import TeachingHistoryNode
from kms.node.teaching.interaction import TeachingInteractionNode
from kms.node.teaching.interaction_tools import TeachingInteractionToolsNode
from kms.node.teaching.knowledge import TeachingKnowledgeNode
from kms.node.teaching.observation import TeachingObservationNode
from kms.node.teaching.persistence import TeachingPersistenceNode
from kms.node.teaching.teacher import TeachingTeacherNode


class TeachingGraph:
    """Compile the complete teaching workflow once."""

    def __init__(
        self,
        history: TeachingHistoryNode,
        interaction: TeachingInteractionNode,
        prober: TeachingTeacherNode,
        planner: TeachingTeacherNode,
        teacher: TeachingTeacherNode,
        tools: TeachingInteractionToolsNode,
        observations: TeachingObservationNode,
        knowledge: TeachingKnowledgeNode,
        persistence: TeachingPersistenceNode,
        researcher: ResearcherNode | None,
        research_tools: ResearchToolsNode | None,
        postresearch: Callable[[TeachingTurnState], object],
        *,
        max_tool_calls: int,
        max_research_tool_calls: int,
    ) -> None:
        graph = StateGraph(TeachingTurnState)
        graph.add_node('teaching_history', history.run)
        graph.add_node('teaching_interaction', interaction.run)
        graph.add_node('teaching_observations', observations.run)
        graph.add_node('teaching_knowledge', knowledge.run)
        graph.add_node('teaching_persistence', persistence.run)
        graph.add_edge(START, 'teaching_history')
        graph.add_edge('teaching_history', 'teaching_interaction')
        add_teaching_decision_phase(
            graph,
            prober,
            planner,
            teacher,
            tools,
            max_tool_calls=max_tool_calls,
        )

        def research_unavailable(
            state: TeachingTurnState,
        ) -> dict[str, object]:
            raise RuntimeError(
                'Research is disabled for this teaching workflow'
            )

        graph.add_node(
            'research_decide',
            researcher.run if researcher is not None else research_unavailable,
        )
        graph.add_node(
            'research_tools',
            research_tools.run
            if research_tools is not None
            else research_unavailable,
        )
        graph.add_node('research_postresearch', postresearch)

        def research_next(state: TeachingTurnState) -> str:
            action = state.latest_action
            assert action is not None
            return (
                'research_postresearch'
                if isinstance(action, FinishResearchAction)
                else 'research_tools'
            )

        graph.add_conditional_edges(
            'research_decide',
            research_next,
            {
                'research_postresearch': 'research_postresearch',
                'research_tools': 'research_tools',
            },
        )
        graph.add_edge('research_tools', 'research_decide')
        graph.add_conditional_edges(
            'research_postresearch',
            lambda state: {
                TeachingPhase.PROBING: 'teaching_probe',
                TeachingPhase.PLANNING: 'teaching_plan',
                TeachingPhase.TEACHING: 'teaching_teach',
            }[state.request.snapshot.session.phase],
            {
                'teaching_probe': 'teaching_probe',
                'teaching_plan': 'teaching_plan',
                'teaching_teach': 'teaching_teach',
            },
        )
        graph.add_edge('teaching_finish', 'teaching_observations')
        graph.add_edge('teaching_observations', 'teaching_knowledge')
        graph.add_edge('teaching_knowledge', 'teaching_persistence')
        graph.add_edge('teaching_persistence', END)
        self._graph: CompiledStateGraph = graph.compile()
        self._recursion_limit = (
            2 * max_tool_calls * (max_research_tool_calls + 2) + 16
        )

    async def run(self, state: TeachingTurnState) -> TeachingTurnState:
        """Run one local-first interaction, optional research, and commit."""
        result = await self._graph.ainvoke(
            state,
            config={'recursion_limit': self._recursion_limit},
        )
        return TeachingTurnState.model_validate(result)
