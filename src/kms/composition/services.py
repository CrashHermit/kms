"""Dependency composition for database-backed application services."""

from kms.config.source_search import SourceSearchSettings
from kms.core.embedding import Embedder
from kms.core.reranking import Reranker
from kms.database.client import DatabaseClient
from kms.database.knowledge.repository import KnowledgeRepository
from kms.database.learning.repository import LearningRepository
from kms.database.source.source_block_repository import SourceBlockRepository
from kms.database.source.source_catalog_repository import (
    SourceCatalogRepository,
)
from kms.database.teaching.observation_repository import (
    LearnerObservationRepository,
)
from kms.database.teaching.repository import TeachingRepository
from kms.database.user.user_repository import UserRepository
from kms.service.knowledge import KnowledgeService
from kms.service.knowledge_search import KnowledgeSearchService
from kms.service.learner_observation_search import (
    LearnerObservationSearchService,
)
from kms.service.learner_observations import LearnerObservationService
from kms.service.learning import LearningService
from kms.service.search_selection import SearchSelectionService
from kms.service.source_evidence import (
    SourceEvidenceService,
    SourceSearchService,
)
from kms.service.teaching import TeachingService
from kms.service.user import UserService


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
    embedder: Embedder,
    reranker: Reranker,
    settings: SourceSearchSettings,
    selection: SearchSelectionService,
) -> SourceSearchService:
    """Compose source search with the invocation's inference clients."""
    return SourceSearchService(
        SourceBlockRepository(database.session),
        embedder,
        reranker,
        settings,
        selection=selection,
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
    embedder: Embedder,
    reranker: Reranker,
    settings: SourceSearchSettings,
    selection: SearchSelectionService,
) -> KnowledgeSearchService:
    """Compose user-owned proposition retrieval with shared inference clients."""
    return KnowledgeSearchService(
        KnowledgeRepository(database.session),
        embedder,
        reranker,
        settings,
        selection=selection,
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
    embedder: Embedder,
    reranker: Reranker,
    settings: SourceSearchSettings,
    selection: SearchSelectionService,
) -> LearnerObservationSearchService:
    """Compose observation retrieval with invocation-scoped inference clients."""
    return LearnerObservationSearchService(
        LearnerObservationRepository(database.session),
        embedder,
        reranker,
        settings,
        selection=selection,
    )
