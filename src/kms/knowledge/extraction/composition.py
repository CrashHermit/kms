"""Compose the unit knowledge extraction node."""

from kms.config.settings import Settings
from kms.database.client import DatabaseClient
from kms.inference.predictors import (
    PredictorFactory,
    build_token_budget,
    resolve_image_budget,
)
from kms.inference.tokenization.resources import LanguageModelTokenizers
from kms.knowledge.extraction.examples import extraction_examples
from kms.knowledge.extraction.inference import (
    UnitExtractionModule,
    UnitExtractionSignature,
)
from kms.knowledge.extraction.node import UnitKnowledgeExtractionNode
from kms.source.retrieval.composition import build_source_reader


def build_node(
    settings: Settings,
    database: DatabaseClient,
    *,
    predictors: PredictorFactory,
    tokenizers: LanguageModelTokenizers,
) -> UnitKnowledgeExtractionNode:
    stage = settings.knowledge_extraction
    examples = extraction_examples()
    budget = build_token_budget(
        settings,
        tokenizers,
        stage.inference,
        input_token_budget=stage.input_token_budget,
        safety_margin_tokens=stage.safety_margin_tokens,
        signature=UnitExtractionSignature,
        examples=examples,
    )
    image_budget = resolve_image_budget(settings, stage.inference)
    predictor = predictors.create(
        UnitExtractionModule,
        stage.inference,
        UnitExtractionSignature,
        examples=examples,
        budget=budget,
        image_budget=image_budget,
    )
    return UnitKnowledgeExtractionNode(
        build_source_reader(database),
        UnitExtractionModule(predictor),
        budget,
        image_budget,
        before_tokens=stage.before_tokens,
        after_tokens=stage.after_tokens,
    )
