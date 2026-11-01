"""Register the bounded teaching decision loop and evidence handoff."""

from typing import cast

from langgraph.graph import END, START, StateGraph

from kms2.core.model.teaching.inference import EvidenceToolAction
from kms2.core.model.teaching.operation import RespondAction
from kms2.core.model.teaching.session import TeachingPhase
from kms2.langgraph.teaching.interaction_state import TeachingInteractionState
from kms2.node.teaching.interaction_tools import TeachingInteractionToolsNode
from kms2.node.teaching.teacher import TeachingTeacherNode

_EVIDENCE_ACTIONS = {
    'search_sources',
    'read_passages',
    'read_images',
    'search_propositions',
    'read_propositions',
    'read_reasoning_steps',
    'search_statements',
    'read_statements',
    'search_procedures',
    'read_procedures',
    'focus_evidence',
    'list_acquired_blocks',
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

    def phase_node(state: TeachingInteractionState) -> str:
        return phase_nodes[state.request.turn.snapshot.session.phase]

    def next_node(state: TeachingInteractionState) -> str:
        action = state.latest_action
        assert action is not None
        if isinstance(action, RespondAction):
            return 'teaching_finish'
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
    graph.add_conditional_edges(START, phase_node, phase_edges)
    destinations = {
        'teaching_evidence': 'teaching_evidence',
        'teaching_finish': 'teaching_finish',
    }
    for node in phase_edges.values():
        graph.add_conditional_edges(node, next_node, destinations)
    graph.add_conditional_edges('teaching_evidence', phase_node, phase_edges)
    graph.add_edge('teaching_finish', END)
