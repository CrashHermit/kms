"""Compose the knowledge preparation workflow."""

from kms.config.settings import Settings
from kms.database.client import DatabaseClient
from kms.inference.embedding.resources import EmbeddingResources
from kms.inference.predictors import PredictorFactory
from kms.inference.tokenization.resources import LanguageModelTokenizers
from kms.knowledge.commit import KnowledgeCommitNode
from kms.knowledge.composition import build_committer
from kms.knowledge.extraction import composition as extraction_composition
from kms.knowledge.preparation.graph import KnowledgePreparationGraph
from kms.knowledge.preparation.node import SourceSearchNode
from kms.knowledge.reasoning import composition as reasoning_composition
from kms.knowledge.retrieval.composition import build_search
from kms.knowledge.retrieval.node import KnowledgeSearchNode
from kms.knowledge.sufficiency import composition as sufficiency_composition
from kms.retrieval.selection.composition import build_search_selector
from kms.source.retrieval.composition import build_source_search_service


def build_knowledge_preparation(
    settings: Settings,
    database: DatabaseClient,
    *,
    embeddings: EmbeddingResources,
    predictors: PredictorFactory,
    tokenizers: LanguageModelTokenizers,
) -> KnowledgePreparationGraph:
    """Build the graph with invocation-owned resources."""
    selector = build_search_selector(
        settings, predictors=predictors, tokenizers=tokenizers
    )
    knowledge_search = KnowledgeSearchNode(
        build_search(
            settings, database, embeddings=embeddings, selection=selector
        )
    )
    committer = KnowledgeCommitNode(
        build_committer(
            settings,
            database,
            embeddings=embeddings,
            predictors=predictors,
            tokenizers=tokenizers,
        )
    )
    source_search = SourceSearchNode(
        build_source_search_service(
            database,
            embedder=embeddings.embedder,
            settings=settings.source_search,
            selection=selector,
        )
    )
    return KnowledgePreparationGraph(
        knowledge_search,
        source_search,
        extraction_composition.build_node(
            settings, database, predictors=predictors, tokenizers=tokenizers
        ),
        reasoning_composition.build_node(
            settings, predictors=predictors, tokenizers=tokenizers
        ),
        sufficiency_composition.build_node(
            settings, predictors=predictors, tokenizers=tokenizers
        ),
        committer,
        source_limit=settings.source_search.result_limit,
    )
