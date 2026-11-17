"""Compose the learning planner and its storage dependency."""

from kms.config.settings import Settings
from kms.database.client import DatabaseClient
from kms.inference.predictors import (
    PredictorFactory,
    build_token_budget,
    resolve_image_budget,
)
from kms.inference.tokenization.resources import LanguageModelTokenizers
from kms.learning.session.composition import build_session_repository
from kms.learning.session.planning.examples import learning_planning_examples
from kms.learning.session.planning.inference import (
    LearningPlanningModule,
    LearningPlanningSignature,
)
from kms.learning.session.planning.node import LearningPlanningNode


def build_node(
    settings: Settings,
    database: DatabaseClient,
    *,
    predictors: PredictorFactory,
    tokenizers: LanguageModelTokenizers,
) -> LearningPlanningNode:
    """Build planning inference with its configured budgets and repository."""
    stage = settings.learning_planning
    examples = learning_planning_examples()
    budget = build_token_budget(
        settings,
        tokenizers,
        stage.inference,
        input_token_budget=stage.input_token_budget,
        safety_margin_tokens=stage.safety_margin_tokens,
        signature=LearningPlanningSignature,
        examples=examples,
    )
    image_budget = resolve_image_budget(settings, stage.inference)
    predictor = predictors.create(
        LearningPlanningModule,
        stage.inference,
        LearningPlanningSignature,
        examples=examples,
        budget=budget,
        image_budget=image_budget,
    )
    return LearningPlanningNode(
        LearningPlanningModule(predictor),
        build_session_repository(settings, database),
    )
