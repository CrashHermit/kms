"""Composition for the shared final search-selection service."""

from kms.composition.predictors import PredictorFactory, build_token_budget
from kms.composition.tokenization import LanguageModelTokenizers
from kms.config.settings import Settings
from kms.core.media_budget import ImagePayloadBudget
from kms.core.tokenization.budget import TokenBudget
from kms.module.search.examples.selection import search_selection_examples
from kms.module.search.selection import (
    SearchSelectionModule,
    SearchSelectionSignature,
)
from kms.service.search_selection import SearchSelectionService


def build_search_selection_service(
    settings: Settings,
    *,
    predictors: PredictorFactory,
    tokenizers: LanguageModelTokenizers,
) -> SearchSelectionService:
    """Compose the selector using application-owned model and tokenizer resources."""
    selection = settings.search_selection
    predictor = predictors.create(
        SearchSelectionModule,
        selection.inference,
        SearchSelectionSignature,
        examples=search_selection_examples(),
    )
    budget: TokenBudget = build_token_budget(
        settings,
        tokenizers,
        selection.inference,
        input_token_budget=selection.input_token_budget,
        safety_margin_tokens=selection.safety_margin_tokens,
    )
    return SearchSelectionService(
        SearchSelectionModule(predictor),
        budget,
        image_budget=ImagePayloadBudget(**selection.image_payload.model_dump()),
    )
