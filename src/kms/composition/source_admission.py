"""Composition for researched source snapshot admission."""

from kms.composition.embedding import EmbeddingResources
from kms.composition.predictors import PredictorFactory
from kms.composition.source_processing import build_source_processing_graph
from kms.composition.tokenization import LanguageModelTokenizers
from kms.config.settings import Settings
from kms.database.client import DatabaseClient
from kms.database.source.source_snapshot_repository import (
    SourceSnapshotRepository,
)
from kms.local_models.runtime import LocalModelRuntime
from kms.service.source_admission import SourceAdmissionService
from kms.service.source_artifacts import SourceArtifactStore
from kms.train.recorder import Recorder


def build_source_admission_service(
    settings: Settings,
    local_models: LocalModelRuntime,
    database: DatabaseClient,
    *,
    embeddings: EmbeddingResources,
    tokenizers: LanguageModelTokenizers,
    predictors: PredictorFactory,
    recorder: Recorder | None = None,
) -> SourceAdmissionService:
    """Compose source processing and snapshot admission services."""
    artifacts = SourceArtifactStore(settings.source_processing.source_directory)
    processing_graph = build_source_processing_graph(
        settings,
        local_models,
        database,
        artifacts=artifacts,
        embeddings=embeddings,
        tokenizers=tokenizers,
        recorder=recorder,
        predictors=predictors,
        persist=False,
    )
    return SourceAdmissionService(
        SourceSnapshotRepository(database.session),
        artifacts,
        processing_graph,
        embedding_model=embeddings.model.model,
        embedding_dimension=embeddings.model.dimension,
    )
