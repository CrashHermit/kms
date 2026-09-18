"""LangGraph registration for instruction discovery."""

from kms2.node.source.instruction_finder import InstructionFinderNode
from langgraph.graph import StateGraph


def add_instruction_finder_phase(
    graph: StateGraph, node: InstructionFinderNode
) -> None:
    """Register the sequential instruction finder node."""
    graph.add_node('instruction_finder', node.run)
    graph.add_edge('exercise_splitter_collect', 'instruction_finder')
