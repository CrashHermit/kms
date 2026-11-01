"""Isolated, tool-bounded LangGraph for one research request."""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from kms2.core.model.research.inference import (
    FinishResearchAction,
    ResearchResult,
    ResearchTask,
)
from kms2.langgraph.research.state import ResearchState
from kms2.node.research.researcher import ResearcherNode
from kms2.node.research.tools import ResearchToolsNode

_TOOL_KINDS = (
    'web_search',
    'fetch_web',
    'list_evidence',
    'read_evidence',
    'open_pdf',
    'search_pdf',
    'read_pdf_pages',
)


class ResearchGraph:
    """Compile a bounded decision/tool loop without persistence nodes."""

    def __init__(
        self,
        researcher: ResearcherNode,
        tools: ResearchToolsNode,
        max_tool_calls: int,
        catalog_page_size: int,
    ) -> None:
        self._researcher = researcher
        self._tools = tools
        self._max_tool_calls = max_tool_calls
        self._catalog_page_size = catalog_page_size
        self.graph = StateGraph(ResearchState)
        self._compiled: CompiledStateGraph | None = None

    def build_graph(self) -> CompiledStateGraph:
        """Compile the typed action loop with an explicit recursion bound."""
        self.graph.add_node('research_decide', self._researcher.run)
        self.graph.add_node('research_tools', self._tools.run)
        self.graph.add_edge(START, 'research_decide')
        self.graph.add_conditional_edges(
            'research_decide',
            self._next_node,
            {
                **{kind: 'research_tools' for kind in _TOOL_KINDS},
                'finish': END,
            },
        )
        self.graph.add_edge('research_tools', 'research_decide')
        self._compiled = self.graph.compile()
        return self._compiled

    async def run(self, task: ResearchTask) -> ResearchResult:
        """Execute one teacher-written task and return selected private evidence."""
        state = ResearchState(
            task=task,
            max_tool_calls=self._max_tool_calls,
            catalog_page_size=self._catalog_page_size,
        )
        result = await self.ainvoke(state)
        if result.final_result is None:
            raise RuntimeError('Research graph ended without a final result')
        return result.final_result

    async def ainvoke(self, state: ResearchState) -> ResearchState:
        """Run one state with recursion limit derived from its tool cap."""
        compiled = self._compiled or self.build_graph()
        result = await compiled.ainvoke(
            state,
            config={'recursion_limit': 2 * self._max_tool_calls + 6},
        )
        return ResearchState.model_validate(result)

    def _next_node(self, state: ResearchState) -> str:
        """Route typed actions and reject any action beyond the hard cap."""
        action = state.latest_action
        assert action is not None
        if isinstance(action, FinishResearchAction):
            return 'finish'
        if state.executed_tool_calls >= self._max_tool_calls:
            raise RuntimeError(
                'Researcher requested a tool after the configured tool-call limit'
            )
        return action.kind
