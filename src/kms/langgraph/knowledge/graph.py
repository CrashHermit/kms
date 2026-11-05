"""Assemble proposition extraction and identity preparation phases."""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from kms.langgraph.knowledge.extraction import add_extraction_phase
from kms.langgraph.knowledge.identity import add_identity_phase
from kms.langgraph.knowledge.state import KnowledgeState
from kms.node.knowledge.extraction import KnowledgeExtractionNode
from kms.node.knowledge.identity import KnowledgeIdentityNode


class KnowledgeAdmissionGraph:
    """Compile the knowledge pipeline once and run typed state values."""

    def __init__(
        self,
        extraction: KnowledgeExtractionNode,
        identity: KnowledgeIdentityNode,
    ) -> None:
        graph = StateGraph(KnowledgeState)
        add_extraction_phase(graph, extraction)
        add_identity_phase(graph, identity)
        graph.add_edge(START, 'knowledge_extraction')
        graph.add_edge('knowledge_identity', END)
        self._graph: CompiledStateGraph = graph.compile()

    async def run(self, state: KnowledgeState) -> KnowledgeState:
        result = await self._graph.ainvoke(state)
        return KnowledgeState.model_validate(result)
