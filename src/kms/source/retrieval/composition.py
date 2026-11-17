"""Compose source retrieval services from application-owned resources."""

from kms.core.embedding import Embedder
from kms.database.client import DatabaseClient
from kms.retrieval.selection.selector import SearchSelector
from kms.source.retrieval.read import SourceReader
from kms.source.retrieval.search import SourceSearchService
from kms.source.retrieval.settings import SourceSearchSettings
from kms.source.storage.retrieval import SourceRetrievalRepository


def build_source_search_service(
    database: DatabaseClient,
    *,
    embedder: Embedder,
    settings: SourceSearchSettings,
    selection: SearchSelector,
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
