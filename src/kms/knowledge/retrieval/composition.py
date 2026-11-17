"""Construct knowledge retrieval from invocation-owned resources."""

from kms.config.settings import Settings
from kms.database.client import DatabaseClient
from kms.inference.embedding.resources import EmbeddingResources
from kms.knowledge.retrieval.search import KnowledgeSearch
from kms.knowledge.storage import KnowledgeRepository
from kms.retrieval.selection.selector import SearchSelector


def build_search(
    settings: Settings,
    database: DatabaseClient,
    *,
    embeddings: EmbeddingResources,
    selection: SearchSelector,
) -> KnowledgeSearch:
    return KnowledgeSearch(
        KnowledgeRepository(database.session),
        embeddings.embedder,
        embeddings.model.model,
        selection=selection,
        settings=settings.knowledge_search,
    )
