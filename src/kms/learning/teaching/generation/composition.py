"""Compose grounded teaching generation and session persistence."""

from kms.config.settings import Settings
from kms.database.client import DatabaseClient
from kms.inference.predictors import (
    PredictorFactory,
    build_token_budget,
    resolve_image_budget,
)
from kms.inference.tokenization.resources import LanguageModelTokenizers
from kms.learning.session.composition import build_session_repository
from kms.learning.teaching.generation.examples import (
    teaching_generation_examples,
)
from kms.learning.teaching.generation.inference import (
    TeachingGenerationModule,
    TeachingGenerationSignature,
)
from kms.learning.teaching.generation.node import TeachingGenerationNode


def build_node(
    settings: Settings,
    database: DatabaseClient,
    *,
    predictors: PredictorFactory,
    tokenizers: LanguageModelTokenizers,
) -> TeachingGenerationNode:
    """Build teaching generation with aligned prompt and image budgets."""
    stage = settings.teaching_generation
    examples = teaching_generation_examples()
    budget = build_token_budget(
        settings,
        tokenizers,
        stage.inference,
        input_token_budget=stage.input_token_budget,
        safety_margin_tokens=stage.safety_margin_tokens,
        signature=TeachingGenerationSignature,
        examples=examples,
    )
    image_budget = resolve_image_budget(settings, stage.inference)
    predictor = predictors.create(
        TeachingGenerationModule,
        stage.inference,
        TeachingGenerationSignature,
        examples=examples,
        budget=budget,
        image_budget=image_budget,
    )
    return TeachingGenerationNode(
        TeachingGenerationModule(predictor),
        build_session_repository(settings, database),
    )
