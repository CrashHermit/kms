from kms.config.settings import Settings
from kms.inference.predictors import PredictorFactory, build_token_budget
from kms.inference.tokenization.resources import LanguageModelTokenizers
from kms.media.budget import ImagePayloadBudget
from kms.source.processing.image_filter.examples import image_filter_examples
from kms.source.processing.image_filter.inference import (
    ImageFilterModule,
    ImageFilterSignature,
)
from kms.source.processing.image_filter.node import ImageFilterNode


def build_node(
    settings: Settings,
    *,
    predictors: PredictorFactory,
    tokenizers: LanguageModelTokenizers,
) -> ImageFilterNode:
    source = settings.source_processing.image_filter
    examples = image_filter_examples()
    budget = build_token_budget(
        settings,
        tokenizers,
        source.inference,
        input_token_budget=source.input_token_budget,
        safety_margin_tokens=source.safety_margin_tokens,
        signature=ImageFilterSignature,
        examples=examples,
    )
    image_budget = ImagePayloadBudget(**source.image_payload.model_dump())
    module = ImageFilterModule(
        predictors.create(
            ImageFilterModule,
            source.inference,
            ImageFilterSignature,
            examples=examples,
            budget=budget,
            image_budget=image_budget,
        )
    )
    return ImageFilterNode(
        module,
        source.context_window,
        budget=budget,
        image_budget=image_budget,
        batch_size=source.batch_size,
        token_counters=tokenizers.counters(source.inference),
    )
