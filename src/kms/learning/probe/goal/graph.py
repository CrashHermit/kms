"""Interactive goal-clarification LangGraph."""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from kms.learning.probe.goal.input import GoalInputNode
from kms.learning.probe.goal.node import GoalClarificationNode
from kms.learning.probe.goal.state import GoalClarificationState


def build_goal_graph(
    input_node: GoalInputNode,
    goal_node: GoalClarificationNode,
) -> CompiledStateGraph:
    """Compile the goal subgraph to inherit its parent graph's checkpointer."""
    graph = StateGraph(GoalClarificationState)
    graph.add_node('input', input_node.run)
    graph.add_node('interpret_goal', goal_node.run)
    graph.add_edge(START, 'input')
    graph.add_edge('input', 'interpret_goal')
    graph.add_conditional_edges(
        'interpret_goal',
        lambda state: state.decision.kind,
        {'clarify': 'input', 'ready': END},
    )
    return graph.compile()
