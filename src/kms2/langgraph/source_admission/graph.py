"""Assembly for research source admission into unpublished snapshots."""

from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from kms2.core.model.source_processing.admission import (
    ResearchSourceMaterial,
    SourceAdmissionResult,
)
from kms2.langgraph.source_admission.finalization import (
    add_finalization_phase,
)
from kms2.langgraph.source_admission.preparation import add_preparation_phase
from kms2.langgraph.source_admission.processing import add_processing_phase
from kms2.langgraph.source_admission.state import SourceAdmissionState
from kms2.node.source_admission.finalization import (
    SourceAdmissionFinalizationNode,
)
from kms2.node.source_admission.preparation import (
    SourceAdmissionPreparationNode,
)
from kms2.node.source_admission.processing import SourceAdmissionProcessingNode


class SourceAdmissionGraph:
    """Compile source-admission phases and execute one material at a time."""

    def __init__(
        self,
        preparation: SourceAdmissionPreparationNode,
        processing: SourceAdmissionProcessingNode,
        finalization: SourceAdmissionFinalizationNode,
    ) -> None:
        graph = StateGraph(SourceAdmissionState)
        add_preparation_phase(graph, preparation)
        add_processing_phase(graph, processing)
        add_finalization_phase(graph, finalization)
        graph.add_edge(START, 'preparation')
        self._graph: CompiledStateGraph = graph.compile()

    async def run(
        self,
        user_uuid: str,
        material: ResearchSourceMaterial,
    ) -> SourceAdmissionResult:
        """Reuse an owned snapshot or return newly processed staged material."""
        result = await self._graph.ainvoke(
            {'user_uuid': user_uuid, 'material': material},
            config={'recursion_limit': 16},
        )
        return result['final_snapshot']
