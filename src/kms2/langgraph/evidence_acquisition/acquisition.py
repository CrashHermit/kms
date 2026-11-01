"""Register the single action-execution node in the evidence graph."""

from langgraph.graph import END, START, StateGraph

from kms2.node.evidence_acquisition.acquisition import EvidenceAcquisitionNode


def add_evidence_acquisition_phase(
    graph: StateGraph,
    acquisition: EvidenceAcquisitionNode,
) -> None:
    """Add the bounded single-action handler between graph boundaries."""
    graph.add_node('evidence_acquisition', acquisition.run)
    graph.add_edge(START, 'evidence_acquisition')
    graph.add_edge('evidence_acquisition', END)
