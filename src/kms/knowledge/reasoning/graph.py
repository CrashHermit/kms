"""Search for knowledge, reason from it, and commit generated candidates."""

from langgraph.graph import END, START, StateGraph

from kms.knowledge.commit import KnowledgeCommitNode
from kms.knowledge.models import ReasonedKnowledge
from kms.knowledge.reasoning.node import KnowledgeReasoningNode
from kms.knowledge.retrieval.node import KnowledgeSearchNode
from kms.knowledge.state import KnowledgeState


class KnowledgeReasoningGraph:
    """Run the standalone search, reasoning, and commit workflow."""

    def __init__(
        self,
        search: KnowledgeSearchNode,
        reasoning: KnowledgeReasoningNode,
        commit: KnowledgeCommitNode,
    ) -> None:
        graph = StateGraph(KnowledgeState)
        graph.add_node('search', search.run)
        graph.add_node('reason', reasoning.run)
        graph.add_node('commit', commit.run)
        graph.add_edge(START, 'search')
        graph.add_edge('search', 'reason')
        graph.add_edge('reason', 'commit')
        graph.add_edge('commit', END)
        self._graph = graph.compile()

    async def reason(
        self, user_uuid: str, understanding: str
    ) -> ReasonedKnowledge:
        """Search, generate, and commit knowledge for one understanding."""
        result = await self._graph.ainvoke(
            {'user_uuid': user_uuid, 'understanding': understanding}
        )
        return ReasonedKnowledge(
            context=result['retrieved_context'], stored=result['stored']
        )
