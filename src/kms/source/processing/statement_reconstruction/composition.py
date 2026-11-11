"""Construction for source statement reconstruction."""

from kms.config.settings import Settings
from kms.inference.predictors import PredictorFactory, build_token_budget
from kms.inference.tokenization.resources import LanguageModelTokenizers
from kms.media.budget import ImagePayloadBudget
from kms.source.processing.statement_reconstruction.examples import (
    source_statement_reconstruction_examples,
)
from kms.source.processing.statement_reconstruction.inference import (
    SourceStatementReconstructionModule,
    SourceStatementReconstructionSignature,
)
from kms.source.processing.statement_reconstruction.node import (
    SourceStatementReconstructionNode,
)


def build_node(
    settings: Settings,
    *,
    predictors: PredictorFactory,
    tokenizers: LanguageModelTokenizers,
) -> SourceStatementReconstructionNode:
    source = settings.source_processing
    stage = source.source_statement_reconstruction
    image_budget = ImagePayloadBudget(**stage.image_payload.model_dump())
    examples = source_statement_reconstruction_examples()
    budget = build_token_budget(
        settings,
        tokenizers,
        stage.inference,
        input_token_budget=stage.input_token_budget,
        safety_margin_tokens=stage.safety_margin_tokens,
        signature=SourceStatementReconstructionSignature,
        examples=examples,
    )
    module = SourceStatementReconstructionModule(
        predictors.create(
            SourceStatementReconstructionModule,
            stage.inference,
            SourceStatementReconstructionSignature,
            examples=examples,
            budget=budget,
            image_budget=image_budget,
        )
    )
    return SourceStatementReconstructionNode(
        module,
        budget=budget,
        batch_size=stage.batch_size,
        image_budget=image_budget,
    )
