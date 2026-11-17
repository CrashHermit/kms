"""Prepare knowledge for one target using owned sources and the knowledge graph."""

from langgraph.graph import END, START, StateGraph

from kms.knowledge.commit import KnowledgeCommitNode
from kms.knowledge.extraction.node import UnitKnowledgeExtractionNode
from kms.knowledge.preparation.models import PreparationResult
from kms.knowledge.preparation.node import SourceSearchNode
from kms.knowledge.reasoning.node import KnowledgeReasoningNode
from kms.knowledge.retrieval.node import KnowledgeSearchNode
from kms.knowledge.state import KnowledgeState
from kms.knowledge.sufficiency.node import KnowledgeSufficiencyNode


class KnowledgePreparationGraph:
    """Search, reason, and extract until sufficient or source hits are exhausted."""

    def __init__(
        self,
        search: KnowledgeSearchNode,
        source_search: SourceSearchNode,
        extraction: UnitKnowledgeExtractionNode,
        reasoning: KnowledgeReasoningNode,
        sufficiency: KnowledgeSufficiencyNode,
        commit: KnowledgeCommitNode,
        *,
        source_limit: int,
    ) -> None:
        self._recursion_limit = 6 * source_limit + 10
        graph = StateGraph(KnowledgeState)
        graph.add_node('search', search.run)
        graph.add_node('judge', sufficiency.run)
        graph.add_node('reason', reasoning.run)
        graph.add_node('search_sources', source_search.run)
        graph.add_node('select_source', source_search.select_next)
        graph.add_node('extract', extraction.run)
        graph.add_node('commit_reasoning', commit.run)
        graph.add_node('commit_extraction', commit.run)
        graph.add_edge(START, 'search')
        graph.add_edge('search', 'judge')
        graph.add_conditional_edges(
            'judge',
            self._after_judgment,
            {
                'done': END,
                'reason': 'reason',
                'search_sources': 'search_sources',
                'select_source': 'select_source',
            },
        )
        graph.add_edge('reason', 'commit_reasoning')
        graph.add_edge('commit_reasoning', 'judge')
        graph.add_conditional_edges(
            'search_sources',
            self._after_source_search,
            {'select_source': 'select_source', 'done': END},
        )
        graph.add_edge('select_source', 'extract')
        graph.add_edge('extract', 'commit_extraction')
        graph.add_edge('commit_extraction', 'reason')
        self._graph = graph.compile()

    async def prepare(
        self, user_uuid: str, understanding: str
    ) -> PreparationResult:
        """Prepare one target and return the final context and sufficiency."""
        result = await self._graph.ainvoke(
            {
                'user_uuid': user_uuid,
                'understanding': understanding,
                'has_reasoned': False,
                'has_searched_sources': False,
                'source_hits': [],
                'source_index': 0,
            },
            config={'recursion_limit': self._recursion_limit},
        )
        return PreparationResult(
            understanding=understanding,
            context=result['context'],
            judgment=result['judgment'],
        )

    @staticmethod
    def _after_judgment(state: KnowledgeState) -> str:
        if state['judgment'].kind == 'sufficient':
            return 'done'
        if not state['has_searched_sources']:
            return 'search_sources' if state['has_reasoned'] else 'reason'
        if state['source_index'] < len(state['source_hits']):
            return 'select_source'
        return 'done'

    @staticmethod
    def _after_source_search(state: KnowledgeState) -> str:
        return 'select_source' if state['source_hits'] else 'done'
