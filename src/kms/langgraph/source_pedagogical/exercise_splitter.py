"""LangGraph registration for the source exercise splitter phase."""

from langgraph.graph import StateGraph

from kms.node.source_pedagogical.exercise_splitter import ExerciseSplitterNode


def add_exercise_splitter_phase(
    graph: StateGraph, node: ExerciseSplitterNode
) -> None:
    """Register exercise splitter fan-out, worker, and collection."""
    graph.add_node('exercise_splitter_worker', node.worker)
    graph.add_node('exercise_splitter_collect', node.collect)
    graph.add_conditional_edges(
        'source_pedagogical_load',
        node.dispatch,
        ['exercise_splitter_worker', 'exercise_splitter_collect'],
    )
    graph.add_edge('exercise_splitter_worker', 'exercise_splitter_collect')
