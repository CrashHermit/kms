"""Composition for owner-scoped learning session storage."""

from kms.artifacts import ContentAddressedArtifactStore
from kms.config.settings import Settings
from kms.database.client import DatabaseClient
from kms.learning.session.repository import LearningSessionRepository


def build_session_repository(
    settings: Settings, database: DatabaseClient
) -> LearningSessionRepository:
    """Build the session repository from application-owned resources."""
    return LearningSessionRepository(
        database.session,
        ContentAddressedArtifactStore(
            settings.learning_session.artifact_directory
        ),
    )
