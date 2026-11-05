"""Register the bounded teaching decision loop and evidence handoff."""

from typing import cast

from langgraph.graph import StateGraph

from kms.core.model.teaching.inference import EvidenceToolAction
from kms.core.model.teaching.operation import RespondAction
from kms.core.model.teaching.session import TeachingPhase
from kms.langgraph.teaching.state import TeachingTurnState
from kms.node.teaching.interaction_tools import TeachingInteractionToolsNode
from kms.node.teaching.teacher import TeachingTeacherNode

_EVIDENCE_ACTIONS = {
    'search_sources',
    'read_passages',
    'read_images',
    'search_propositions',
    'read_propositions',
    'read_arguments',
    'search_learner_observations',
    'read_learner_observations',
    'search_statements',
    'search_arguments',
    'read_statements',
    'search_procedures',
    'read_procedures',
    'focus_evidence',
    'acquire_evidence',
}


def add_teaching_decision_phase(
    graph: StateGraph,
    prober: TeachingTeacherNode,
    planner: TeachingTeacherNode,
    teacher: TeachingTeacherNode,
    tools: TeachingInteractionToolsNode,
    *,
    max_tool_calls: int,
) -> None:
    """Register phase decisions, bounded evidence execution, and finish."""
    phase_nodes = {
        TeachingPhase.PROBING: 'teaching_probe',
        TeachingPhase.PLANNING: 'teaching_plan',
        TeachingPhase.TEACHING: 'teaching_teach',
    }
    phase_edges = {node: node for node in phase_nodes.values()}

    def phase_node(state: TeachingTurnState) -> str:
        return phase_nodes[state.request.snapshot.session.phase]

    def next_node(state: TeachingTurnState) -> str:
        action = state.latest_action
        assert action is not None
        if isinstance(action, RespondAction):
            error = tools.action_error(state)
            if error is None:
                return 'teaching_finish'
            if state.tool_calls >= max_tool_calls:
                raise RuntimeError(
                    f'Teacher response has invalid references at the tool-call limit: {error}'
                )
            return 'teaching_evidence'
        if state.tool_calls >= max_tool_calls:
            raise RuntimeError(
                'Teacher requested a tool after the configured tool-call limit'
            )
        evidence_action = cast(EvidenceToolAction, action)
        if evidence_action.kind not in _EVIDENCE_ACTIONS:
            raise AssertionError(
                f'Unexpected teaching action kind: {evidence_action.kind}'
            )
        return 'teaching_evidence'

    graph.add_node('teaching_probe', prober.run)
    graph.add_node('teaching_plan', planner.run)
    graph.add_node('teaching_teach', teacher.run)
    graph.add_node('teaching_evidence', tools.run)
    graph.add_node('teaching_finish', tools.finish)
    graph.add_conditional_edges('teaching_interaction', phase_node, phase_edges)
    destinations = {
        'teaching_finish': 'teaching_finish',
        'teaching_evidence': 'teaching_evidence',
    }
    for node in phase_edges.values():
        graph.add_conditional_edges(node, next_node, destinations)

    def after_evidence(state: TeachingTurnState) -> str:
        return (
            'research_decide'
            if state.research_task is not None
            else phase_node(state)
        )

    graph.add_conditional_edges(
        'teaching_evidence',
        after_evidence,
        {**phase_edges, 'research_decide': 'research_decide'},
    )
