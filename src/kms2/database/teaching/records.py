"""Shared hydration for teaching questions and learner observations."""

import json
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from neo4j.time import DateTime

from kms2.core.model.teaching.observation import (
    LearnerObservation,
    LearnerObservationEvidence,
    LearnerObservationRecord,
)
from kms2.core.model.teaching.question import (
    TeachingQuestion,
    TeachingQuestionKind,
    question_view,
)


def question_from_record(row: Mapping[str, Any]) -> TeachingQuestion:
    """Convert one private persisted question specification."""
    return TeachingQuestion(
        uuid=row['uuid'],
        kind=TeachingQuestionKind(row['kind']),
        prompt=row['prompt'],
        response=json.loads(row['response_json']),
        created_at=native_datetime(row['created_at']),
    )


def native_datetime(value: datetime | DateTime) -> datetime:
    """Convert a Neo4j temporal value to the domain datetime type."""
    return value.to_native() if isinstance(value, DateTime) else value


def learner_observation_from_record(
    row: Mapping[str, Any],
) -> LearnerObservationRecord:
    """Hydrate one stored observation with learner-safe question evidence."""
    observation_row = row['observation']
    observation = LearnerObservation(
        uuid=observation_row['uuid'],
        kind=observation_row['kind'],
        topic=observation_row['topic'],
        description=observation_row['description'],
        embedding=observation_row['embedding'],
        created_at=native_datetime(observation_row['created_at']),
    )
    evidence: list[LearnerObservationEvidence] = []
    for item in row['evidence']:
        question_row = item['question']
        evidence.append(
            LearnerObservationEvidence(
                turn_uuid=item['turn_uuid'],
                session_uuid=item['session_uuid'],
                content=item['content'],
                created_at=native_datetime(item['created_at']),
                question=(
                    question_view(question_from_record(question_row))
                    if question_row is not None
                    else None
                ),
            )
        )
    return LearnerObservationRecord(observation=observation, evidence=evidence)
