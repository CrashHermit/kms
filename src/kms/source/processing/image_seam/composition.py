from kms.config.settings import Settings
from kms.inference.predictors import PredictorFactory, build_token_budget
from kms.inference.tokenization.resources import LanguageModelTokenizers
from kms.media.budget import ImagePayloadBudget
from kms.source.processing.image_seam.examples import image_seam_examples
from kms.source.processing.image_seam.inference import (
    ImageSeamJudgeModule,
    ImageSeamSignature,
)
from kms.source.processing.image_seam.node import ImageSeamNode


def build_node(
    settings: Settings,
    *,
    predictors: PredictorFactory,
    tokenizers: LanguageModelTokenizers,
) -> ImageSeamNode:
    source = settings.source_processing.image_seam
    examples = image_seam_examples()
    budget = build_token_budget(
        settings,
        tokenizers,
        source.inference,
        input_token_budget=source.input_token_budget,
        safety_margin_tokens=source.safety_margin_tokens,
        signature=ImageSeamSignature,
        examples=examples,
    )
    image_budget = ImagePayloadBudget(**source.image_payload.model_dump())
    judge = ImageSeamJudgeModule(
        predictors.create(
            ImageSeamJudgeModule,
            source.inference,
            ImageSeamSignature,
            examples=examples,
            budget=budget,
            image_budget=image_budget,
        )
    )
    return ImageSeamNode(
        judge,
        budget=budget,
        image_budget=image_budget,
        batch_size=source.batch_size,
    )
