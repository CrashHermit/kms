"""Registration for serial proposition identity preparation."""

from langgraph.graph import StateGraph

from kms.node.knowledge.identity import KnowledgeIdentityNode


def add_identity_phase(
    graph: StateGraph,
    node: KnowledgeIdentityNode,
) -> None:
    """Register identity preparation as the terminal admission phase."""
    graph.add_node('knowledge_identity', node.run)
