"""Compose grounded task generation and session persistence."""

from kms.config.settings import Settings
from kms.database.client import DatabaseClient
from kms.inference.predictors import (
    PredictorFactory,
    build_token_budget,
    resolve_image_budget,
)
from kms.inference.tokenization.resources import LanguageModelTokenizers
from kms.learning.session.composition import build_session_repository
from kms.learning.task.generation.examples import task_generation_examples
from kms.learning.task.generation.inference import (
    TaskGenerationModule,
    TaskGenerationSignature,
)
from kms.learning.task.generation.node import TaskGenerationNode


def build_node(
    settings: Settings,
    database: DatabaseClient,
    *,
    predictors: PredictorFactory,
    tokenizers: LanguageModelTokenizers,
) -> TaskGenerationNode:
    """Build task generation with aligned prompt and native-image budgets."""
    stage = settings.task_generation
    examples = task_generation_examples()
    budget = build_token_budget(
        settings,
        tokenizers,
        stage.inference,
        input_token_budget=stage.input_token_budget,
        safety_margin_tokens=stage.safety_margin_tokens,
        signature=TaskGenerationSignature,
        examples=examples,
    )
    image_budget = resolve_image_budget(settings, stage.inference)
    predictor = predictors.create(
        TaskGenerationModule,
        stage.inference,
        TaskGenerationSignature,
        examples=examples,
        budget=budget,
        image_budget=image_budget,
    )
    return TaskGenerationNode(
        TaskGenerationModule(predictor),
        build_session_repository(settings, database),
    )
