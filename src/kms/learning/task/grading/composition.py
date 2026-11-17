"""Compose the standalone free-response grading node."""

from kms.config.settings import Settings
from kms.database.client import DatabaseClient
from kms.inference.predictors import (
    PredictorFactory,
    build_token_budget,
    resolve_image_budget,
)
from kms.inference.tokenization.resources import LanguageModelTokenizers
from kms.learning.session.composition import build_session_repository
from kms.learning.task.grading.examples import task_grading_examples
from kms.learning.task.grading.inference import (
    TaskGradingModule,
    TaskGradingSignature,
)
from kms.learning.task.grading.node import TaskGradingNode


def build_node(
    settings: Settings,
    database: DatabaseClient,
    *,
    predictors: PredictorFactory,
    tokenizers: LanguageModelTokenizers,
) -> TaskGradingNode:
    """Build task grading with aligned examples and native-image budgets."""
    stage = settings.task_grading
    examples = task_grading_examples()
    budget = build_token_budget(
        settings,
        tokenizers,
        stage.inference,
        input_token_budget=stage.input_token_budget,
        safety_margin_tokens=stage.safety_margin_tokens,
        signature=TaskGradingSignature,
        examples=examples,
    )
    image_budget = resolve_image_budget(settings, stage.inference)
    predictor = predictors.create(
        TaskGradingModule,
        stage.inference,
        TaskGradingSignature,
        budget=budget,
        image_budget=image_budget,
        examples=examples,
    )
    return TaskGradingNode(
        TaskGradingModule(predictor),
        build_session_repository(settings, database),
    )
