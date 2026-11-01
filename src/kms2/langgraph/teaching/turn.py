"""Register the linear pre-commit stages of one teaching turn."""

from langgraph.graph import START, StateGraph

from kms2.node.teaching.history import TeachingHistoryNode
from kms2.node.teaching.interaction import TeachingInteractionNode
from kms2.node.teaching.knowledge import TeachingKnowledgeNode
from kms2.node.teaching.observation import TeachingObservationNode


def add_teaching_turn_phase(
    graph: StateGraph,
    history: TeachingHistoryNode,
    interaction: TeachingInteractionNode,
    observations: TeachingObservationNode,
    knowledge: TeachingKnowledgeNode,
) -> None:
    """Register history, interaction, observation, and knowledge stages."""
    graph.add_node('teaching_history', history.run)
    graph.add_node('teaching_interaction', interaction.run)
    graph.add_node('teaching_observations', observations.run)
    graph.add_node('teaching_knowledge', knowledge.run)
    graph.add_edge(START, 'teaching_history')
    graph.add_edge('teaching_history', 'teaching_interaction')
    graph.add_edge('teaching_interaction', 'teaching_observations')
    graph.add_edge('teaching_observations', 'teaching_knowledge')
