"""Composition for procedure discovery."""

from kms.config.settings import Settings
from kms.inference.predictors import (
    PredictorFactory,
    build_token_budget,
    resolve_image_budget,
)
from kms.inference.tokenization.resources import LanguageModelTokenizers
from kms.source.processing.procedure_finder.examples import (
    procedure_boundary_examples,
    procedure_start_examples,
)
from kms.source.processing.procedure_finder.inference import (
    ProcedureBoundaryRouterModule,
    ProcedureBoundaryRouterSignature,
    ProcedureStartRouterModule,
    ProcedureStartRouterSignature,
)
from kms.source.processing.procedure_finder.node import ProcedureFinderNode


def build_node(
    settings: Settings,
    *,
    predictors: PredictorFactory,
    tokenizers: LanguageModelTokenizers,
) -> ProcedureFinderNode:
    stage = settings.source_processing.procedure_finder
    image_budget = resolve_image_budget(
        settings, stage.start_router, stage.boundary_router
    )
    start_examples, boundary_examples = (
        procedure_start_examples(),
        procedure_boundary_examples(),
    )
    start_budget = build_token_budget(
        settings,
        tokenizers,
        stage.start_router,
        input_token_budget=stage.start_input_token_budget,
        safety_margin_tokens=stage.safety_margin_tokens,
        signature=ProcedureStartRouterSignature,
        examples=start_examples,
    )
    boundary_budget = build_token_budget(
        settings,
        tokenizers,
        stage.boundary_router,
        input_token_budget=stage.boundary_input_token_budget,
        safety_margin_tokens=stage.safety_margin_tokens,
        signature=ProcedureBoundaryRouterSignature,
        examples=boundary_examples,
    )
    start = ProcedureStartRouterModule(
        predictors.create(
            ProcedureStartRouterModule,
            stage.start_router,
            ProcedureStartRouterSignature,
            examples=start_examples,
            budget=start_budget,
            image_budget=image_budget,
        )
    )
    boundary = ProcedureBoundaryRouterModule(
        predictors.create(
            ProcedureBoundaryRouterModule,
            stage.boundary_router,
            ProcedureBoundaryRouterSignature,
            examples=boundary_examples,
            budget=boundary_budget,
            image_budget=image_budget,
        )
    )
    return ProcedureFinderNode(
        start,
        boundary,
        stage,
        start_budget=start_budget,
        boundary_budget=boundary_budget,
        start_token_counters=tokenizers.counters(stage.start_router),
        boundary_token_counters=tokenizers.counters(stage.boundary_router),
        image_budget=image_budget,
    )
