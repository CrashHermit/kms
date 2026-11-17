"""Compose FSRS review behavior with owner-scoped persistence."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fsrs import Scheduler

from kms.database.client import DatabaseClient
from kms.learning.memory.node import MemoryReviewNode
from kms.learning.memory.repository import MemoryRepository

if TYPE_CHECKING:
    from kms.config.settings import Settings


def build_node(
    settings: Settings, database: DatabaseClient
) -> MemoryReviewNode:
    """Build memory review with configured FSRS and repository dependencies."""
    return MemoryReviewNode(
        Scheduler(
            desired_retention=settings.memory.desired_retention,
            enable_fuzzing=False,
        ),
        MemoryRepository(database.session),
    )
