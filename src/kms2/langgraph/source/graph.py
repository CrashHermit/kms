"""Source-processing LangGraph assembly for KMS2."""

from kms2.langgraph.source.content_correction import (
    add_content_correction_phase,
)
from kms2.langgraph.source.formatting import add_formatter_phase
from kms2.langgraph.source.image_enrichment import add_image_enrichment_phase
from kms2.langgraph.source.image_seam import add_image_seam_phase
from kms2.langgraph.source.ocr import add_ocr_phase
from kms2.langgraph.source.persistence import add_persistence_phase
from kms2.langgraph.source.splitter import add_splitter_phase
from kms2.langgraph.source.state import SourceState
from kms2.langgraph.source.text_seam import add_text_seam_phase
from kms2.node.source.content_correction import ContentCorrectionNode
from kms2.node.source.formatting import FormattingNode
from kms2.node.source.image_enrichment import ImageEnrichmentNode
from kms2.node.source.image_seam import ImageSeamNode
from kms2.node.source.ocr import OCRNode
from kms2.node.source.persistence import SourcePersistenceNode
from kms2.node.source.splitter import SplitterNode
from kms2.node.source.text_seam import TextSeamNode
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph


class SourceGraph:
    """Compile the source-processing graph from injected node behaviors."""

    def __init__(
        self,
        ocr: OCRNode,
        content_correction: ContentCorrectionNode,
        formatter: FormattingNode,
        text_seam: TextSeamNode,
        image_seam: ImageSeamNode,
        image_enrichment: ImageEnrichmentNode,
        splitter: SplitterNode,
        persistence: SourcePersistenceNode,
    ) -> None:
        self.graph = StateGraph(SourceState)
        self.ocr = ocr
        self.content_correction = content_correction
        self.formatter = formatter
        self.text_seam = text_seam
        self.image_seam = image_seam
        self.image_enrichment = image_enrichment
        self.splitter = splitter
        self.persistence = persistence

    def build_graph(self) -> CompiledStateGraph:
        """Register source phases and compile the graph."""
        add_ocr_phase(self.graph, self.ocr)
        add_content_correction_phase(self.graph, self.content_correction)
        add_formatter_phase(self.graph, self.formatter)
        add_text_seam_phase(self.graph, self.text_seam)
        add_image_seam_phase(self.graph, self.image_seam)
        add_image_enrichment_phase(self.graph, self.image_enrichment)
        add_splitter_phase(self.graph, self.splitter)
        add_persistence_phase(self.graph, self.persistence)
        return self.graph.compile()
