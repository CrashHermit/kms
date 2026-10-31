"""Dependency composition for database-backed application services."""

from kms2.database.client import DatabaseClient
from kms2.database.learning.repository import LearningRepository
from kms2.database.source.source_catalog_repository import (
    SourceCatalogRepository,
)
from kms2.database.teaching.repository import TeachingRepository
from kms2.database.user.user_repository import UserRepository
from kms2.service.learning import LearningService
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
