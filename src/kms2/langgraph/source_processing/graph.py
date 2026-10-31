"""Source-processing LangGraph assembly for KMS2."""

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from kms2.langgraph.source_processing.content_correction import (
    add_content_correction_phase,
)
from kms2.langgraph.source_processing.embedding import add_embedding_phase
from kms2.langgraph.source_processing.formatting import add_formatter_phase
from kms2.langgraph.source_processing.image_description import (
    add_image_description_phase,
)
from kms2.langgraph.source_processing.image_seam import add_image_seam_phase
from kms2.langgraph.source_processing.ocr import add_ocr_phase
from kms2.langgraph.source_processing.persistence import add_persistence_phase
from kms2.langgraph.source_processing.state import SourceProcessingState
from kms2.langgraph.source_processing.text_seam import add_text_seam_phase
from kms2.node.source_processing.content_correction import ContentCorrectionNode
from kms2.node.source_processing.embedding import EmbeddingNode
from kms2.node.source_processing.formatting import FormattingNode
from kms2.node.source_processing.image_description import ImageDescriptionNode
from kms2.node.source_processing.image_seam import ImageSeamNode
from kms2.node.source_processing.ocr import OCRNode
from kms2.node.source_processing.persistence import SourcePersistenceNode
from kms2.node.source_processing.text_seam import TextSeamNode


class SourceProcessingGraph:
    """Compile the source-processing graph from injected node behaviors."""

    def __init__(
        self,
        ocr: OCRNode,
        content_correction: ContentCorrectionNode,
        formatter: FormattingNode,
        text_seam: TextSeamNode,
        image_seam: ImageSeamNode,
        image_description: ImageDescriptionNode,
        embedding: EmbeddingNode,
        persistence: SourcePersistenceNode,
    ) -> None:
        self.graph = StateGraph(SourceProcessingState)
        self.ocr = ocr
        self.content_correction = content_correction
        self.formatter = formatter
        self.text_seam = text_seam
        self.image_seam = image_seam
        self.image_description = image_description
        self.embedding = embedding
        self.persistence = persistence

    def build_graph(self) -> CompiledStateGraph:
        """Register source phases and compile the graph."""
        add_ocr_phase(self.graph, self.ocr)
        add_content_correction_phase(self.graph, self.content_correction)
        add_formatter_phase(self.graph, self.formatter)
        add_text_seam_phase(self.graph, self.text_seam)
        add_image_seam_phase(self.graph, self.image_seam)
        add_image_description_phase(self.graph, self.image_description)
        add_embedding_phase(self.graph, self.embedding)
        add_persistence_phase(self.graph, self.persistence)
        self.graph.add_edge('persistence', END)
        return self.graph.compile()
