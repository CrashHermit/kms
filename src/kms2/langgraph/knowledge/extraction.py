"""Registration for grounded proposition extraction."""

from langgraph.graph import StateGraph

from kms2.node.knowledge.extraction import KnowledgeExtractionNode


def add_extraction_phase(
    graph: StateGraph,
    node: KnowledgeExtractionNode,
) -> None:
    """Register extraction as the first admission-graph phase."""
    graph.add_node('knowledge_extraction', node.run)
    graph.add_edge('knowledge_extraction', 'knowledge_identity')
