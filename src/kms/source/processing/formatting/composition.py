from kms.config.settings import Settings
from kms.inference.predictors import PredictorFactory, build_token_budget
from kms.inference.tokenization.resources import LanguageModelTokenizers
from kms.source.processing.formatting.examples import formatting_examples
from kms.source.processing.formatting.inference import (
    FormatterModule,
    FormatterSignature,
)
from kms.source.processing.formatting.node import FormattingNode


def build_node(
    settings: Settings,
    *,
    predictors: PredictorFactory,
    tokenizers: LanguageModelTokenizers,
) -> FormattingNode:
    source = settings.source_processing.formatting
    examples = formatting_examples()
    budget = build_token_budget(
        settings,
        tokenizers,
        source.inference,
        input_token_budget=source.input_token_budget,
        safety_margin_tokens=source.safety_margin_tokens,
        signature=FormatterSignature,
        examples=examples,
    )
    formatter = FormatterModule(
        predictors.create(
            FormatterModule,
            source.inference,
            FormatterSignature,
            examples=examples,
            budget=budget,
        )
    )
    return FormattingNode(
        formatter,
        budget=budget,
        batch_size=source.batch_size,
    )
