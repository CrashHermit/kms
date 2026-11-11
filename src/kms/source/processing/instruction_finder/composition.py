"""Construction for instruction discovery."""

from kms.config.settings import Settings
from kms.inference.predictors import PredictorFactory, build_token_budget
from kms.inference.tokenization.resources import LanguageModelTokenizers
from kms.media.budget import ImagePayloadBudget
from kms.source.processing.instruction_finder.examples import (
    instruction_boundary_examples,
    instruction_start_examples,
)
from kms.source.processing.instruction_finder.inference import (
    InstructionBoundaryRouterModule,
    InstructionBoundaryRouterSignature,
    InstructionStartRouterModule,
    InstructionStartRouterSignature,
)
from kms.source.processing.instruction_finder.node import InstructionFinderNode


def build_node(
    settings: Settings,
    *,
    predictors: PredictorFactory,
    tokenizers: LanguageModelTokenizers,
) -> InstructionFinderNode:
    source = settings.source_processing
    stage = source.instruction_finder
    image_budget = ImagePayloadBudget(**stage.image_payload.model_dump())
    start_examples = instruction_start_examples()
    start_budget = build_token_budget(
        settings,
        tokenizers,
        stage.start_router,
        input_token_budget=stage.start_input_token_budget,
        safety_margin_tokens=stage.safety_margin_tokens,
        signature=InstructionStartRouterSignature,
        examples=start_examples,
    )
    boundary_examples = instruction_boundary_examples()
    boundary_budget = build_token_budget(
        settings,
        tokenizers,
        stage.boundary_router,
        input_token_budget=stage.boundary_input_token_budget,
        safety_margin_tokens=stage.safety_margin_tokens,
        signature=InstructionBoundaryRouterSignature,
        examples=boundary_examples,
    )
    start = InstructionStartRouterModule(
        predictors.create(
            InstructionStartRouterModule,
            stage.start_router,
            InstructionStartRouterSignature,
            examples=start_examples,
            budget=start_budget,
            image_budget=image_budget,
        )
    )
    boundary = InstructionBoundaryRouterModule(
        predictors.create(
            InstructionBoundaryRouterModule,
            stage.boundary_router,
            InstructionBoundaryRouterSignature,
            examples=boundary_examples,
            budget=boundary_budget,
            image_budget=image_budget,
        )
    )
    return InstructionFinderNode(
        start,
        boundary,
        stage,
        start_budget=start_budget,
        boundary_budget=boundary_budget,
        start_token_counters=tokenizers.counters(stage.start_router),
        boundary_token_counters=tokenizers.counters(stage.boundary_router),
        image_budget=image_budget,
    )
