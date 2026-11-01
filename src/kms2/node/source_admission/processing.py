"""Run ordinary source processing without persistence."""

from langgraph.graph.state import CompiledStateGraph

from kms2.core.model.page import SourcePage
from kms2.langgraph.source_admission.state import SourceAdmissionState


class SourceAdmissionProcessingNode:
    """Process prepared material using the shared source graph."""

    def __init__(self, processing_graph: CompiledStateGraph) -> None:
        self._processing_graph = processing_graph

    async def run(
        self, state: SourceAdmissionState
    ) -> dict[str, list[SourcePage]]:
        preparation = state.preparation
        assert preparation is not None
        result = await self._processing_graph.ainvoke(
            {'source': preparation.source, 'input': preparation.input}
        )
        return {'processed_pages': result['embedded_pages']}
