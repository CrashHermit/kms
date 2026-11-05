"""Source-processing LangGraph assembly for KMS2."""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from kms.langgraph.source_processing.content_correction import (
    add_content_correction_phase,
)
from kms.langgraph.source_processing.embedding import add_embedding_phase
from kms.langgraph.source_processing.formatting import add_formatter_phase
from kms.langgraph.source_processing.image_description import (
    add_image_description_phase,
)
from kms.langgraph.source_processing.image_filter import add_image_filter_phase
from kms.langgraph.source_processing.image_seam import add_image_seam_phase
from kms.langgraph.source_processing.ocr import add_ocr_phase
from kms.langgraph.source_processing.persistence import add_persistence_phase
from kms.langgraph.source_processing.state import SourceProcessingState
from kms.langgraph.source_processing.text_seam import add_text_seam_phase
from kms.node.source_processing.content_correction import ContentCorrectionNode
from kms.node.source_processing.embedding import EmbeddingNode
from kms.node.source_processing.formatting import FormattingNode
from kms.node.source_processing.image_description import ImageDescriptionNode
from kms.node.source_processing.image_filter import ImageFilterNode
from kms.node.source_processing.image_seam import ImageSeamNode
from kms.node.source_processing.ocr import OCRNode
from kms.node.source_processing.persistence import SourcePersistenceNode
from kms.node.source_processing.text_seam import TextSeamNode


class SourceProcessingGraph:
    """Compile the source-processing graph from injected node behaviors."""

    def __init__(
        self,
        ocr: OCRNode,
        content_correction: ContentCorrectionNode,
        formatter: FormattingNode,
        text_seam: TextSeamNode,
        image_seam: ImageSeamNode,
        image_filter: ImageFilterNode,
        image_description: ImageDescriptionNode,
        embedding: EmbeddingNode,
        persistence: SourcePersistenceNode,
        *,
        persist: bool = True,
    ) -> None:
        graph = StateGraph(SourceProcessingState)
        add_ocr_phase(graph, ocr)
        graph.add_edge(START, 'ocr')
        add_content_correction_phase(graph, content_correction)
        add_formatter_phase(graph, formatter)
        add_text_seam_phase(graph, text_seam)
        add_image_seam_phase(graph, image_seam)
        add_image_filter_phase(graph, image_filter)
        add_image_description_phase(graph, image_description)
        add_embedding_phase(graph, embedding)
        if persist:
            add_persistence_phase(graph, persistence)
            graph.add_edge('persistence', END)
        else:
            graph.add_edge('embedding', END)
        self._graph: CompiledStateGraph = graph.compile()

    async def run(self, state: SourceProcessingState) -> SourceProcessingState:
        # The longest route is OCR + eight dispatch/worker/collect phases
        # + embedding + optional persistence: 27 node supersteps. Keep a
        # separate bound because this graph also runs nested under research.
        result = await self._graph.ainvoke(
            state, config={'recursion_limit': 28}
        )
        return SourceProcessingState.model_validate(result)
