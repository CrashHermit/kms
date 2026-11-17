"""Compose the knowledge sufficiency node."""

from kms.config.settings import Settings
from kms.inference.predictors import PredictorFactory, build_token_budget
from kms.inference.tokenization.resources import LanguageModelTokenizers
from kms.knowledge.sufficiency.inference import (
    SufficiencyModule,
    SufficiencySignature,
)
from kms.knowledge.sufficiency.node import KnowledgeSufficiencyNode


def build_node(
    settings: Settings,
    *,
    predictors: PredictorFactory,
    tokenizers: LanguageModelTokenizers,
) -> KnowledgeSufficiencyNode:
    stage = settings.knowledge_sufficiency
    budget = build_token_budget(
        settings,
        tokenizers,
        stage.inference,
        input_token_budget=stage.input_token_budget,
        safety_margin_tokens=stage.safety_margin_tokens,
        signature=SufficiencySignature,
    )
    module = SufficiencyModule(
        predictors.create(
            SufficiencyModule,
            stage.inference,
            SufficiencySignature,
            budget=budget,
        )
    )
    return KnowledgeSufficiencyNode(module)
