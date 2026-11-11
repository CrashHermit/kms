"""Compose the goal-clarification graph and its inference."""

from langgraph.graph.state import CompiledStateGraph

from kms.config.settings import Settings
from kms.inference.predictors import PredictorFactory
from kms.learning.probe.goal.graph import build_goal_graph
from kms.learning.probe.goal.inference import (
    GoalClarificationModule,
    GoalClarificationSignature,
)
from kms.learning.probe.goal.input import GoalInputNode
from kms.learning.probe.goal.node import GoalClarificationNode


def build_goal_clarification_graph(
    settings: Settings,
    *,
    predictors: PredictorFactory,
) -> CompiledStateGraph:
    """Build the interactive graph with configured text inference."""
    predictor = predictors.create(
        GoalClarificationModule,
        settings.goal_clarification.inference,
        GoalClarificationSignature,
    )
    return build_goal_graph(
        GoalInputNode(),
        GoalClarificationNode(GoalClarificationModule(predictor)),
    )
