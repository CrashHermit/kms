"""Compile and run one bounded teaching decision loop."""

from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from kms2.core.model.teaching.workflow import (
    TeachingInteractionRequest,
    TeachingInteractionResult,
)
from kms2.langgraph.teaching.interaction import add_teaching_decision_phase
from kms2.langgraph.teaching.interaction_state import TeachingInteractionState
from kms2.node.teaching.interaction_tools import TeachingInteractionToolsNode
from kms2.node.teaching.teacher import TeachingTeacherNode
from kms2.service.teaching_workspace import restore_workspace


class TeachingInteractionGraph:
    """Run phase decisions and tool actions until one response is produced."""

    def __init__(
        self,
        prober: TeachingTeacherNode,
        planner: TeachingTeacherNode,
        teacher: TeachingTeacherNode,
        tools: TeachingInteractionToolsNode,
        *,
        max_tool_calls: int,
    ) -> None:
        graph = StateGraph(TeachingInteractionState)
        add_teaching_decision_phase(
            graph,
            prober,
            planner,
            teacher,
            tools,
            max_tool_calls=max_tool_calls,
        )
        self._compiled: CompiledStateGraph = graph.compile()
        self._recursion_limit = 2 * max_tool_calls + 6

    async def run(
        self, request: TeachingInteractionRequest
    ) -> TeachingInteractionResult:
        """Invoke the compiled loop once and return its completed response."""
        workspace = await restore_workspace(
            request.turn.snapshot.workspace_json
        )
        result = await self._compiled.ainvoke(
            TeachingInteractionState(request=request, workspace=workspace),
            config={'recursion_limit': self._recursion_limit},
        )
        state = TeachingInteractionState.model_validate(result)
        assert state.result is not None
        return state.result
