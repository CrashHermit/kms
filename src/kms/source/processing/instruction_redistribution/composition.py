"""Composition for instruction detection and redistribution."""

from kms.config.settings import Settings
from kms.inference.predictors import (
    PredictorFactory,
    build_token_budget,
    resolve_image_budget,
)
from kms.inference.tokenization.resources import LanguageModelTokenizers
from kms.source.processing.instruction_redistribution.examples import (
    instruction_applicability_examples,
    instruction_boundary_examples,
    instruction_start_examples,
)
from kms.source.processing.instruction_redistribution.inference import (
    InstructionApplicabilityModule,
    InstructionApplicabilitySignature,
    InstructionBoundaryRouterModule,
    InstructionBoundaryRouterSignature,
    InstructionStartRouterModule,
    InstructionStartRouterSignature,
)
from kms.source.processing.instruction_redistribution.node import (
    InstructionRedistributionNode,
)


def build_node(
    settings: Settings,
    *,
    predictors: PredictorFactory,
    tokenizers: LanguageModelTokenizers,
) -> InstructionRedistributionNode:
    stage = settings.source_processing.instruction_redistribution
    detection, applicability = stage.detection, stage.applicability
    detection_budget_image = resolve_image_budget(
        settings, detection.start_router, detection.boundary_router
    )
    applicability_image = resolve_image_budget(
        settings, applicability.inference
    )
    start_examples = instruction_start_examples()
    boundary_examples = instruction_boundary_examples()
    applicability_examples = instruction_applicability_examples()
    start_budget = build_token_budget(
        settings,
        tokenizers,
        detection.start_router,
        input_token_budget=detection.start_input_token_budget,
        safety_margin_tokens=detection.safety_margin_tokens,
        signature=InstructionStartRouterSignature,
        examples=start_examples,
    )
    boundary_budget = build_token_budget(
        settings,
        tokenizers,
        detection.boundary_router,
        input_token_budget=detection.boundary_input_token_budget,
        safety_margin_tokens=detection.safety_margin_tokens,
        signature=InstructionBoundaryRouterSignature,
        examples=boundary_examples,
    )
    applicability_budget = build_token_budget(
        settings,
        tokenizers,
        applicability.inference,
        input_token_budget=applicability.input_token_budget,
        safety_margin_tokens=applicability.safety_margin_tokens,
        signature=InstructionApplicabilitySignature,
        examples=applicability_examples,
    )
    start = InstructionStartRouterModule(
        predictors.create(
            InstructionStartRouterModule,
            detection.start_router,
            InstructionStartRouterSignature,
            examples=start_examples,
            budget=start_budget,
            image_budget=detection_budget_image,
        )
    )
    boundary = InstructionBoundaryRouterModule(
        predictors.create(
            InstructionBoundaryRouterModule,
            detection.boundary_router,
            InstructionBoundaryRouterSignature,
            examples=boundary_examples,
            budget=boundary_budget,
            image_budget=detection_budget_image,
        )
    )
    judge = InstructionApplicabilityModule(
        predictors.create(
            InstructionApplicabilityModule,
            applicability.inference,
            InstructionApplicabilitySignature,
            examples=applicability_examples,
            budget=applicability_budget,
            image_budget=applicability_image,
        )
    )
    return InstructionRedistributionNode(
        start,
        boundary,
        judge,
        stage,
        start_budget=start_budget,
        boundary_budget=boundary_budget,
        applicability_budget=applicability_budget,
        start_token_counters=tokenizers.counters(detection.start_router),
        boundary_token_counters=tokenizers.counters(detection.boundary_router),
        applicability_token_counters=tokenizers.counters(
            applicability.inference
        ),
        image_budget=detection_budget_image,
        applicability_image_budget=applicability_image,
    )
