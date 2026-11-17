"""Compose the shared retrieval selector from application-owned resources."""

from kms.config.settings import Settings
from kms.core.tokenization.budget import TokenBudget
from kms.inference.predictors import (
    PredictorFactory,
    build_token_budget,
    resolve_image_budget,
)
from kms.inference.tokenization.resources import LanguageModelTokenizers
from kms.retrieval.selection.examples import search_selection_examples
from kms.retrieval.selection.inference import (
    SearchSelectionModule,
    SearchSelectionSignature,
)
from kms.retrieval.selection.selector import SearchSelector


def build_search_selector(
    settings: Settings,
    *,
    predictors: PredictorFactory,
    tokenizers: LanguageModelTokenizers,
) -> SearchSelector:
    """Compose the selector using application-owned model and tokenizer resources."""
    selection = settings.search_selection
    examples = search_selection_examples()
    budget: TokenBudget = build_token_budget(
        settings,
        tokenizers,
        selection.inference,
        input_token_budget=selection.input_token_budget,
        safety_margin_tokens=selection.safety_margin_tokens,
        signature=SearchSelectionSignature,
        examples=examples,
    )
    image_budget = resolve_image_budget(settings, selection.inference)
    predictor = predictors.create(
        SearchSelectionModule,
        selection.inference,
        SearchSelectionSignature,
        examples=examples,
        budget=budget,
        image_budget=image_budget,
    )
    return SearchSelector(
        SearchSelectionModule(predictor),
        budget,
        image_budget=image_budget,
    )
