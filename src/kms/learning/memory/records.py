"""Hydration for memory and target records read from Neo4j."""

from datetime import UTC
from typing import Any

from pydantic import TypeAdapter

from kms.knowledge.models import Argument, Proposition
from kms.learning.memory.models import MemoryReview, MemoryState
from kms.learning.references import LearningTarget
from kms.learning.session.records import native_datetime
from kms.learning.task.models import ReviewRating


def hydrate_memory(record: dict[str, Any], target: Any) -> MemoryState:
    """Hydrate durable card fields with target identity from its relationship."""
    return MemoryState(
        uuid=record['uuid'],
        target=TypeAdapter(LearningTarget).validate_python(dict(target)),
        card_json=record['card_json'],
        due_at=native_datetime(record['due_at']).astimezone(UTC),
        review_count=record['review_count'],
        created_at=native_datetime(record['created_at']),
        updated_at=native_datetime(record['updated_at']),
        invalidated_at=(
            native_datetime(record['invalidated_at'])
            if record['invalidated_at'] is not None
            else None
        ),
    )


def hydrate_review(record: dict[str, Any], memory_uuid: str) -> MemoryReview:
    """Hydrate a review with assessment identity from its provenance edge."""
    return MemoryReview(
        uuid=record['uuid'],
        memory_uuid=memory_uuid,
        assessment_uuid=record['assessment_uuid'],
        rating=ReviewRating(record['rating']),
        reviewed_at=native_datetime(record['reviewed_at']).astimezone(UTC),
        sequence=record['sequence'],
        created_at=native_datetime(record['created_at']),
        updated_at=native_datetime(record['updated_at']),
        invalidated_at=(
            native_datetime(record['invalidated_at'])
            if record['invalidated_at'] is not None
            else None
        ),
    )


def hydrate_proposition(record: dict[str, Any]) -> Proposition:
    """Hydrate a proposition from its owned graph record."""
    return Proposition(
        uuid=record['uuid'],
        name=record['name'],
        statement=record['statement'],
        created_at=native_datetime(record['created_at']),
        updated_at=native_datetime(record['updated_at']),
        invalidated_at=(
            native_datetime(record['invalidated_at'])
            if record['invalidated_at'] is not None
            else None
        ),
    )


def hydrate_argument(
    record: dict[str, Any], premises: list[Any], conclusion: Any
) -> Argument:
    """Hydrate an owned argument using its relationship-owned endpoints."""
    premise_records = [dict(premise) for premise in premises]
    return Argument(
        uuid=record['uuid'],
        premise_uuids=sorted(premise['uuid'] for premise in premise_records),
        conclusion_uuid=conclusion['uuid'],
        created_at=native_datetime(record['created_at']),
        updated_at=native_datetime(record['updated_at']),
        invalidated_at=(
            native_datetime(record['invalidated_at'])
            if record['invalidated_at'] is not None
            else None
        ),
    )
