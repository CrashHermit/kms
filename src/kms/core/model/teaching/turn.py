"""Teaching conversation turn graph models."""

from datetime import datetime
from enum import StrEnum

from kms.core.model.base import Vertex


class TeachingRole(StrEnum):
    """Role that authored one teaching turn."""

    LEARNER = 'learner'
    TEACHER = 'teacher'


class TeachingTurn(Vertex):
    """One immutable learner or teacher message in a teaching session."""

    role: TeachingRole
    content: str
    created_at: datetime
