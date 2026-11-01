"""Composition for the explicit database-only legacy source finalizer."""

from kms2.database.client import DatabaseClient
from kms2.database.source.research_source_migration_repository import (
    ResearchSourceMigrationRepository,
)
from kms2.service.research_source_migration import (
    ResearchSourceMigrationFinalizationService,
)


def build_research_source_migration_finalizer(
    database: DatabaseClient,
) -> ResearchSourceMigrationFinalizationService:
    """Build the database-only global legacy cleanup operation."""
    return ResearchSourceMigrationFinalizationService(
        ResearchSourceMigrationRepository(database.session)
    )
