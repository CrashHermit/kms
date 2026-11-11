"""Construction for source procedure reconstruction."""

from kms.config.settings import Settings
from kms.inference.predictors import PredictorFactory, build_token_budget
from kms.inference.tokenization.resources import LanguageModelTokenizers
from kms.media.budget import ImagePayloadBudget
from kms.source.processing.procedure_reconstruction.examples import (
    source_procedure_reconstruction_examples,
)
from kms.source.processing.procedure_reconstruction.inference import (
    SourceProcedureReconstructionModule,
    SourceProcedureReconstructionSignature,
)
from kms.source.processing.procedure_reconstruction.node import (
    SourceProcedureReconstructionNode,
)


def build_node(
    settings: Settings,
    *,
    predictors: PredictorFactory,
    tokenizers: LanguageModelTokenizers,
) -> SourceProcedureReconstructionNode:
    source = settings.source_processing
    stage = source.source_procedure_reconstruction
    image_budget = ImagePayloadBudget(**stage.image_payload.model_dump())
    examples = source_procedure_reconstruction_examples()
    budget = build_token_budget(
        settings,
        tokenizers,
        stage.inference,
        input_token_budget=stage.input_token_budget,
        safety_margin_tokens=stage.safety_margin_tokens,
        signature=SourceProcedureReconstructionSignature,
        examples=examples,
    )
    module = SourceProcedureReconstructionModule(
        predictors.create(
            SourceProcedureReconstructionModule,
            stage.inference,
            SourceProcedureReconstructionSignature,
            examples=examples,
            budget=budget,
            image_budget=image_budget,
        )
    )
    return SourceProcedureReconstructionNode(
        module,
        budget=budget,
        batch_size=stage.batch_size,
        image_budget=image_budget,
    )
