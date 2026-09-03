"""LangGraph coordination for the KMS2 source-processing pipeline."""

from dataclasses import dataclass

from kms2.langgraph.source.state import SourceState
from kms2.node.source.content_correction import ContentCorrectorNode
from kms2.ocr.mistral import 
from langgraph.graph import END, START, StateGraph


class SourceGraph:
    def build_nodes(self):

    def build_source_graph(self) -> CompiledStateGraph:
        """Build and compile the ordered source-processing graph.
        
        The graph keeps provider-specific work inside the supplied nodes. Its
        coordination boundary is provider-neutral: OCR produces artifacts
        consumed by the content corrector, which produces the terminal state.
        """
        graph = StateGraph(SourceState)
        graph.add_node('ocr', )
        graph.add_node('content_corrector', nodes.content_corrector)
        graph.add_edge(START, 'ocr')
        graph.add_edge('ocr', 'content_corrector')
        graph.add_edge('content_corrector', END)
        return graph.compile()
