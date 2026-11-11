"""Construction for pedagogical discovery."""

from kms.config.settings import Settings
from kms.inference.predictors import PredictorFactory, build_token_budget
from kms.inference.tokenization.resources import LanguageModelTokenizers
from kms.media.budget import ImagePayloadBudget
from kms.source.processing.pedagogical_finder.examples import (
    pedagogical_boundary_examples,
    pedagogical_start_examples,
)
from kms.source.processing.pedagogical_finder.inference import (
    PedagogicalBoundaryRouterModule,
    PedagogicalBoundaryRouterSignature,
    PedagogicalStartRouterModule,
    PedagogicalStartRouterSignature,
)
from kms.source.processing.pedagogical_finder.node import PedagogicalFinderNode


def build_node(
    settings: Settings,
    *,
    predictors: PredictorFactory,
    tokenizers: LanguageModelTokenizers,
) -> PedagogicalFinderNode:
    source = settings.source_processing
    stage = source.pedagogical_finder
    image_budget = ImagePayloadBudget(**stage.image_payload.model_dump())
    start_examples = pedagogical_start_examples()
    start_budget = build_token_budget(
        settings,
        tokenizers,
        stage.start_router,
        input_token_budget=stage.start_input_token_budget,
        safety_margin_tokens=stage.safety_margin_tokens,
        signature=PedagogicalStartRouterSignature,
        examples=start_examples,
    )
    boundary_examples = pedagogical_boundary_examples()
    boundary_budget = build_token_budget(
        settings,
        tokenizers,
        stage.boundary_router,
        input_token_budget=stage.boundary_input_token_budget,
        safety_margin_tokens=stage.safety_margin_tokens,
        signature=PedagogicalBoundaryRouterSignature,
        examples=boundary_examples,
    )
    start = PedagogicalStartRouterModule(
        predictors.create(
            PedagogicalStartRouterModule,
            stage.start_router,
            PedagogicalStartRouterSignature,
            examples=start_examples,
            budget=start_budget,
            image_budget=image_budget,
        )
    )
    boundary = PedagogicalBoundaryRouterModule(
        predictors.create(
            PedagogicalBoundaryRouterModule,
            stage.boundary_router,
            PedagogicalBoundaryRouterSignature,
            examples=boundary_examples,
            budget=boundary_budget,
            image_budget=image_budget,
        )
    )
    return PedagogicalFinderNode(
        start,
        boundary,
        stage,
        start_budget=start_budget,
        boundary_budget=boundary_budget,
        start_token_counters=tokenizers.counters(stage.start_router),
        boundary_token_counters=tokenizers.counters(stage.boundary_router),
        image_budget=image_budget,
    )
