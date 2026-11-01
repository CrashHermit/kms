"""Assemble proposition extraction and identity preparation phases."""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from kms2.core.model.knowledge.admission import (
    KnowledgeAdmission,
    KnowledgeExtractionInput,
)
from kms2.core.model.source import SourceRevisionReference
from kms2.langgraph.knowledge.extraction import add_extraction_phase
from kms2.langgraph.knowledge.identity import add_identity_phase
from kms2.langgraph.knowledge.state import KnowledgeState
from kms2.node.knowledge.extraction import KnowledgeExtractionNode
from kms2.node.knowledge.identity import KnowledgeIdentityNode


class KnowledgeAdmissionGraph:
    """Prepare grounded knowledge records without opening a write transaction."""

    def __init__(
        self,
        extraction: KnowledgeExtractionNode,
        identity: KnowledgeIdentityNode,
    ) -> None:
        self._extraction = extraction
        self._identity = identity
        self.graph = StateGraph(KnowledgeState)
        self._compiled_graph = self.build_graph()

    def build_graph(self) -> CompiledStateGraph:
        """Compile the extraction then identity preparation pipeline."""
        graph = self.graph
        add_extraction_phase(graph, self._extraction)
        add_identity_phase(graph, self._identity)
        graph.add_edge(START, 'knowledge_extraction')
        graph.add_edge('knowledge_identity', END)
        return graph.compile()

    async def run(
        self,
        user_uuid: str,
        request: KnowledgeExtractionInput,
        *,
        existing_proposition_uuids: list[str],
        supporting_block_uuids: list[str],
        block_source_revisions: list[SourceRevisionReference],
        evidence_image_bytes: list[bytes],
    ) -> KnowledgeAdmission | None:
        """Return the complete admission payload while leaving persistence to caller."""
        result = await self._compiled_graph.ainvoke(
            KnowledgeState(
                user_uuid=user_uuid,
                request=request,
                existing_proposition_uuids=existing_proposition_uuids,
                supporting_block_uuids=supporting_block_uuids,
                block_source_revisions=block_source_revisions,
                evidence_image_bytes=evidence_image_bytes,
            )
        )
        return result['admission']
