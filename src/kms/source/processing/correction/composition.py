from kms.config.settings import Settings
from kms.inference.predictors import PredictorFactory, build_token_budget
from kms.inference.tokenization.resources import LanguageModelTokenizers
from kms.media.budget import ImagePayloadBudget
from kms.source.processing.correction.examples import (
    content_correction_examples,
)
from kms.source.processing.correction.inference import (
    ContentCorrectorModule,
    ContentCorrectorSignature,
)
from kms.source.processing.correction.node import ContentCorrectionNode


def build_node(
    settings: Settings,
    *,
    predictors: PredictorFactory,
    tokenizers: LanguageModelTokenizers,
) -> ContentCorrectionNode:
    source = settings.source_processing.content_correction
    examples = content_correction_examples()
    budget = build_token_budget(
        settings,
        tokenizers,
        source.inference,
        input_token_budget=source.input_token_budget,
        safety_margin_tokens=source.safety_margin_tokens,
        signature=ContentCorrectorSignature,
        examples=examples,
    )
    image_budget = ImagePayloadBudget(**source.image_payload.model_dump())
    corrector = ContentCorrectorModule(
        predictors.create(
            ContentCorrectorModule,
            source.inference,
            ContentCorrectorSignature,
            examples=examples,
            budget=budget,
            image_budget=image_budget,
        )
    )
    return ContentCorrectionNode(
        corrector,
        budget=budget,
        image_budget=image_budget,
        batch_size=source.batch_size,
    )
