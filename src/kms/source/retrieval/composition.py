"""Compose source retrieval services from application-owned resources."""

from kms.config.settings import Settings
from kms.core.embedding import Embedder
from kms.core.tokenization.budget import TokenBudget
from kms.database.client import DatabaseClient
from kms.inference.predictors import PredictorFactory, build_token_budget
from kms.inference.tokenization.resources import LanguageModelTokenizers
from kms.media.budget import ImagePayloadBudget
from kms.source.retrieval.read import SourceReader
from kms.source.retrieval.search import SourceSearchService
from kms.source.retrieval.selection.examples import search_selection_examples
from kms.source.retrieval.selection.inference import (
    SearchSelectionModule,
    SearchSelectionSignature,
)
from kms.source.retrieval.selection.service import SearchSelectionService
from kms.source.retrieval.settings import SourceSearchSettings
from kms.source.storage.retrieval import SourceRetrievalRepository


def build_search_selection_service(
    settings: Settings,
    *,
    predictors: PredictorFactory,
    tokenizers: LanguageModelTokenizers,
) -> SearchSelectionService:
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
    image_budget = ImagePayloadBudget(**selection.image_payload.model_dump())
    predictor = predictors.create(
        SearchSelectionModule,
        selection.inference,
        SearchSelectionSignature,
        examples=examples,
        budget=budget,
        image_budget=image_budget,
    )
    return SearchSelectionService(
        SearchSelectionModule(predictor),
        budget,
        image_budget=image_budget,
    )


def build_source_search_service(
    database: DatabaseClient,
    *,
    embedder: Embedder,
    settings: SourceSearchSettings,
    selection: SearchSelectionService,
) -> SourceSearchService:
    """Compose source search with the invocation's inference clients."""
    return SourceSearchService(
        SourceRetrievalRepository(database.session),
        embedder,
        settings,
        selection=selection,
    )


def build_source_reader(database: DatabaseClient) -> SourceReader:
    """Compose owner-scoped database and image evidence reads."""
    return SourceReader(SourceRetrievalRepository(database.session))
