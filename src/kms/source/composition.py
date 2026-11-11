"""Compose source ingestion with application-owned processing resources."""

from kms.config.settings import Settings
from kms.database.client import DatabaseClient
from kms.inference.embedding.resources import EmbeddingResources
from kms.inference.predictors import PredictorFactory
from kms.inference.tokenization.resources import LanguageModelTokenizers
from kms.source.ingestion import SourceIngestionService
from kms.source.processing.composition import build_source_processing_graph
from kms.source.storage.artifacts import SourceArtifactStore
from kms.source.storage.snapshots import (
    SourceSnapshotRepository,
)


def build_source_ingestion_service(
    settings: Settings,
    database: DatabaseClient,
    *,
    predictors: PredictorFactory,
    embeddings: EmbeddingResources,
    tokenizers: LanguageModelTokenizers,
) -> SourceIngestionService:
    """Compose source processing and complete snapshot ingestion."""
    artifacts = SourceArtifactStore(settings.source_processing.source_directory)
    processing_graph = build_source_processing_graph(
        settings,
        embeddings=embeddings,
        tokenizers=tokenizers,
        predictors=predictors,
    )
    return SourceIngestionService(
        SourceSnapshotRepository(database.session),
        artifacts,
        processing_graph,
        embedding_model=embeddings.model.model,
        embedding_dimension=embeddings.model.dimension,
    )
