"""Durable learner memory and review contracts."""

import json
from datetime import datetime
from typing import TypedDict
from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel, ConfigDict

from kms.core.models import Vertex
from kms.knowledge.models import KnowledgeContext
from kms.learning.references import LearningTarget
from kms.learning.task.models import (
    ReviewRating,
    Task,
)


class MemoryState(Vertex):
    """Current FSRS card state for one owned target."""

    target: LearningTarget
    card_json: str
    due_at: datetime
    review_count: int


class MemoryReview(Vertex):
    """One applied review sourced from a graded assessment."""

    memory_uuid: str
    assessment_uuid: str
    rating: ReviewRating
    reviewed_at: datetime
    sequence: int


class MemoryHistory(BaseModel):
    """A memory card and its actual scheduling events."""

    model_config = ConfigDict(extra='forbid')

    memory: MemoryState
    reviews: list[MemoryReview]


class ReviewTargetContext(BaseModel):
    """Owned material needed to generate a task or teaching for a target."""

    model_config = ConfigDict(extra='forbid')

    target: LearningTarget
    understanding: str
    knowledge: KnowledgeContext
    solution_task: Task | None = None


class ReviewInput(BaseModel):
    """Assessment facts resolved from its owned response and optional memory."""

    model_config = ConfigDict(extra='forbid')

    target: LearningTarget
    response_uuid: str
    assessment_uuid: str
    rating: ReviewRating
    reviewed_at: datetime
    memory: MemoryState | None = None


class MemoryReviewState(TypedDict):
    """State required to apply one graded response to memory."""

    user_uuid: str
    session_uuid: str
    assessment_uuid: str


def memory_uuid(user_uuid: str, target: LearningTarget) -> str:
    """Return the stable UUID for one user and typed learning target."""
    identity = json.dumps(
        [user_uuid, target.kind, target.uuid], separators=(',', ':')
    )
    return str(uuid5(NAMESPACE_URL, f'kms:learning-memory:{identity}'))


def review_uuid(user_uuid: str, response_uuid: str) -> str:
    """Return the stable UUID for one learner response's scheduling event."""
    identity = json.dumps([user_uuid, response_uuid], separators=(',', ':'))
    return str(uuid5(NAMESPACE_URL, f'kms:learning-review:{identity}'))
