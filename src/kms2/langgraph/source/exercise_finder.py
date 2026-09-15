"""LangGraph registration for exercise component discovery."""

from kms2.node.source.exercise_finder import ExerciseFinderNode
from langgraph.graph import StateGraph


def add_exercise_finder_phase(
    graph: StateGraph, node: ExerciseFinderNode
) -> None:
    """Register the sequential exercise component finder."""
    graph.add_node('exercise_finder', node.run)
    graph.add_edge('instruction_finder', 'exercise_finder')
