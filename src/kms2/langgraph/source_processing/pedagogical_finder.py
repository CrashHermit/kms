"""LangGraph registration for pedagogical discovery."""

from langgraph.graph import StateGraph

from kms2.node.source_processing.pedagogical_finder import PedagogicalFinderNode


def add_pedagogical_finder_phase(
    graph: StateGraph, node: PedagogicalFinderNode
) -> None:
    """Register the sequential pedagogical finder node."""
    graph.add_node('pedagogical_finder', node.run)
    graph.add_edge('exercise_finder', 'pedagogical_finder')
