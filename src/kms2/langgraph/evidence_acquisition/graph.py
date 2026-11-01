"""One-action LangGraph boundary for teaching evidence operations."""

from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from kms2.core.model.teaching.workflow import (
    EvidenceExecutionRequest,
    EvidenceExecutionResult,
)
from kms2.langgraph.evidence_acquisition.acquisition import (
    add_evidence_acquisition_phase,
)
from kms2.langgraph.evidence_acquisition.state import EvidenceAcquisitionState
from kms2.node.evidence_acquisition.acquisition import EvidenceAcquisitionNode


class EvidenceAcquisitionGraph:
    """Execute exactly one evidence action with an isolated state graph."""

    def __init__(self, acquisition: EvidenceAcquisitionNode) -> None:
        graph = StateGraph(EvidenceAcquisitionState)
        add_evidence_acquisition_phase(graph, acquisition)
        self._compiled: CompiledStateGraph = graph.compile()

    async def run(
        self, request: EvidenceExecutionRequest
    ) -> EvidenceExecutionResult:
        result = await self._compiled.ainvoke(
            EvidenceAcquisitionState(request=request),
            config={'recursion_limit': 4},
        )
        state = EvidenceAcquisitionState.model_validate(result)
        assert state.result is not None
        return state.result
