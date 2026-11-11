"""Construction for exercise discovery."""

from kms.config.settings import Settings
from kms.inference.predictors import PredictorFactory, build_token_budget
from kms.inference.tokenization.resources import LanguageModelTokenizers
from kms.media.budget import ImagePayloadBudget
from kms.source.processing.exercise_finder.examples import (
    exercise_boundary_examples,
    exercise_start_examples,
)
from kms.source.processing.exercise_finder.inference import (
    ExerciseBoundaryRouterModule,
    ExerciseBoundaryRouterSignature,
    ExerciseStartRouterModule,
    ExerciseStartRouterSignature,
)
from kms.source.processing.exercise_finder.node import ExerciseFinderNode


def build_node(
    settings: Settings,
    *,
    predictors: PredictorFactory,
    tokenizers: LanguageModelTokenizers,
) -> ExerciseFinderNode:
    source = settings.source_processing
    stage = source.exercise_finder
    image_budget = ImagePayloadBudget(**stage.image_payload.model_dump())
    start_examples = exercise_start_examples()
    start_budget = build_token_budget(
        settings,
        tokenizers,
        stage.start_router,
        input_token_budget=stage.start_input_token_budget,
        safety_margin_tokens=stage.safety_margin_tokens,
        signature=ExerciseStartRouterSignature,
        examples=start_examples,
    )
    boundary_examples = exercise_boundary_examples()
    boundary_budget = build_token_budget(
        settings,
        tokenizers,
        stage.boundary_router,
        input_token_budget=stage.boundary_input_token_budget,
        safety_margin_tokens=stage.safety_margin_tokens,
        signature=ExerciseBoundaryRouterSignature,
        examples=boundary_examples,
    )
    start = ExerciseStartRouterModule(
        predictors.create(
            ExerciseStartRouterModule,
            stage.start_router,
            ExerciseStartRouterSignature,
            examples=start_examples,
            budget=start_budget,
            image_budget=image_budget,
        )
    )
    boundary = ExerciseBoundaryRouterModule(
        predictors.create(
            ExerciseBoundaryRouterModule,
            stage.boundary_router,
            ExerciseBoundaryRouterSignature,
            examples=boundary_examples,
            budget=boundary_budget,
            image_budget=image_budget,
        )
    )
    return ExerciseFinderNode(
        start,
        boundary,
        stage,
        start_budget=start_budget,
        boundary_budget=boundary_budget,
        start_token_counters=tokenizers.counters(stage.start_router),
        boundary_token_counters=tokenizers.counters(stage.boundary_router),
        image_budget=image_budget,
    )
