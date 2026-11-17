"""Compose source-grounded correctness inference."""

from kms.config.settings import Settings
from kms.inference.predictors import (
    PredictorFactory,
    build_token_budget,
    resolve_image_budget,
)
from kms.inference.tokenization.resources import LanguageModelTokenizers
from kms.knowledge.correctness.inference import (
    CorrectnessModule,
    CorrectnessSignature,
)


def build_correctness_judge(
    settings: Settings,
    *,
    predictors: PredictorFactory,
    tokenizers: LanguageModelTokenizers,
) -> CorrectnessModule:
    stage = settings.knowledge_correctness
    budget = build_token_budget(
        settings,
        tokenizers,
        stage.inference,
        input_token_budget=stage.input_token_budget,
        safety_margin_tokens=stage.safety_margin_tokens,
        signature=CorrectnessSignature,
    )
    image_budget = resolve_image_budget(settings, stage.inference)
    return CorrectnessModule(
        predictors.create(
            CorrectnessModule,
            stage.inference,
            CorrectnessSignature,
            budget=budget,
            image_budget=image_budget,
        )
    )
