from langgraph.graph.state import CompiledStateGraph

from kms2.langgraph.source.state import SourceState
from kms2.node.source.ocr import OCRNode
from kms2.node.source.content_correction import ContentCorrectorNode

from langgraph.graph import END, START, StateGraph


class SourceGraph:
    def __init__(self) -> None:
        self.graph = StateGraph(SourceState)

    def _build_nodes(self):
        self.graph.add_node('ocr', node=OCRNode)
        self.graph.add_node('content_corrector', node=ContentCorrectorNode)

    def _build_edges(self):
        """Build and compile the ordered source-processing graph.
        
        The graph keeps provider-specific work inside the supplied nodes. Its
        coordination boundary is provider-neutral: OCR produces artifacts
        consumed by the content corrector, which produces the terminal state.
        """
        graph.add_edge(START, 'ocr')
        graph.add_edge('ocr', 'content_corrector')
        graph.add_edge('content_corrector', END)
        return graph.compile()

    def build_graph(self) -> CompiledStateGraph:
        self._build_nodes()
        self._build_edges()

        compiled_state_graph: CompiledStateGraph = self.graph.compile()

        return compiled_state_graph
