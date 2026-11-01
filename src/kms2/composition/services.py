"""Dependency composition for database-backed application services."""

from pathlib import Path

from kms2.config.source_search import SourceSearchSettings
from kms2.core.embedding import EmbeddingClient
from kms2.core.reranking import RerankerClient
from kms2.database.client import DatabaseClient
from kms2.database.knowledge.repository import KnowledgeRepository
from kms2.database.learning.repository import LearningRepository
from kms2.database.source.research_source_migration_repository import (
    ResearchSourceMigrationRepository,
)
from kms2.database.source.source_block_repository import SourceBlockRepository
from kms2.database.source.source_catalog_repository import (
    SourceCatalogRepository,
)
from kms2.database.source.source_snapshot_repository import (
    SourceSnapshotRepository,
)
from kms2.database.teaching.observation_repository import (
    LearnerObservationRepository,
)
from kms2.database.teaching.repository import TeachingRepository
from kms2.database.user.user_repository import UserRepository
from kms2.service.knowledge import KnowledgeService
from kms2.service.knowledge_search import KnowledgeSearchService
from kms2.service.learner_observation_search import (
    LearnerObservationSearchService,
)
from kms2.service.learner_observations import LearnerObservationService
from kms2.service.learning import LearningService
from kms2.service.research_source_migration import (
    ResearchSourceMigrationService,
)
from kms2.service.source_evidence import (
    SourceEvidenceService,
    SourceSearchService,
)
from kms2.service.teaching import TeachingService
from kms2.service.user import UserService


def build_learning_service(database: DatabaseClient) -> LearningService:
    """Compose deck and review workflows without starting inference resources."""
    return LearningService(LearningRepository(database.session))


def build_teaching_service(database: DatabaseClient) -> TeachingService:
    """Compose teaching-session persistence without inference resources."""
    return TeachingService(TeachingRepository(database.session))


def build_user_service(database: DatabaseClient) -> UserService:
    """Compose user provisioning and source ownership workflows."""
    return UserService(
        UserRepository(database.session),
        SourceCatalogRepository(database.session),
    )


def build_source_search_service(
    database: DatabaseClient,
    *,
    embedding_client: EmbeddingClient,
    reranker_client: RerankerClient,
    settings: SourceSearchSettings,
) -> SourceSearchService:
    """Compose source search with the invocation's inference clients."""
    return SourceSearchService(
        SourceBlockRepository(database.session),
        embedding_client,
        reranker_client,
        settings,
    )


def build_source_evidence_service(
    database: DatabaseClient,
) -> SourceEvidenceService:
    """Compose database and image evidence reading without inference clients."""
    return SourceEvidenceService(SourceBlockRepository(database.session))


def build_knowledge_service(database: DatabaseClient) -> KnowledgeService:
    """Compose canonical knowledge reads and admission record preparation."""
    return KnowledgeService(KnowledgeRepository(database.session))


def build_knowledge_search_service(
    database: DatabaseClient,
    *,
    embedding_client: EmbeddingClient,
    reranker_client: RerankerClient,
    settings: SourceSearchSettings,
) -> KnowledgeSearchService:
    """Compose user-owned proposition retrieval with shared inference clients."""
    return KnowledgeSearchService(
        KnowledgeRepository(database.session),
        embedding_client,
        reranker_client,
        settings,
    )


def build_learner_observation_service(
    database: DatabaseClient,
) -> LearnerObservationService:
    """Compose database-only learner observation reads."""
    return LearnerObservationService(
        LearnerObservationRepository(database.session)
    )


def build_learner_observation_search_service(
    database: DatabaseClient,
    *,
    embedding_client: EmbeddingClient,
    reranker_client: RerankerClient,
    settings: SourceSearchSettings,
) -> LearnerObservationSearchService:
    """Compose observation retrieval with invocation-scoped inference clients."""
    return LearnerObservationSearchService(
        LearnerObservationRepository(database.session),
        embedding_client,
        reranker_client,
        settings,
    )


def build_research_source_migration_service(
    database: DatabaseClient,
    *,
    embedding_client: EmbeddingClient,
    storage_root: Path,
) -> ResearchSourceMigrationService:
    """Compose no-network research migration with shared source persistence."""
    return ResearchSourceMigrationService(
        ResearchSourceMigrationRepository(database.session),
        SourceSnapshotRepository(database.session),
        embedding_client,
        storage_root,
    )
