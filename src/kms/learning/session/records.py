"""Serialization, media finalization, and hydration for session records."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, TypeAdapter

from kms.artifacts import ContentAddressedArtifactStore
from kms.learning.references import GroundingReferences, LearningTarget
from kms.learning.session.models import (
    LearningDecisionRecord,
    LearningSession,
    SessionHistory,
    SessionStep,
)
from kms.learning.task.models import (
    Solution,
    Task,
    TaskAnswer,
    TaskAssessment,
    TaskGrade,
    TaskResponse,
)
from kms.learning.teaching.models import Teaching
from kms.media.content import collect_image_assets
from kms.media.models import ImageAsset, ImagePart, Material


def read_payload(value: str | dict[str, Any]) -> dict[str, Any]:
    """Decode payload JSON and restore path-backed native image payloads."""
    data = json.loads(value) if isinstance(value, str) else value
    return _restore_image_paths(data)


def finalize_materials(
    record: BaseModel, artifacts: ContentAddressedArtifactStore
) -> BaseModel:
    """Persist each unique image asset before database publication."""
    assets = collect_image_assets(record)
    finalized = {
        asset.key: ImageAsset(
            key=asset.key,
            payload=artifacts.write(
                ContentAddressedArtifactStore.read(asset.payload)
                if isinstance(asset.payload, Path)
                else asset.payload,
                suffix='.bin',
            ),
        )
        for asset in assets
    }
    return _replace_materials(record, finalized)


def _replace_materials(
    record: BaseModel, assets: dict[tuple[str, ...], ImageAsset]
) -> BaseModel:
    """Copy a model tree while replacing image references by durable paths."""
    updates: dict[str, Any] = {}
    for name, field in type(record).model_fields.items():
        if field.exclude:
            continue
        value = getattr(record, name)
        if isinstance(value, Material):
            updates[name] = Material(
                parts=tuple(
                    ImagePart(asset=assets[part.asset.key])
                    if isinstance(part, ImagePart)
                    else part
                    for part in value.parts
                )
            )
        elif isinstance(value, BaseModel):
            updates[name] = _replace_materials(value, assets)
        elif (
            isinstance(value, list)
            and value
            and isinstance(value[0], BaseModel)
        ):
            updates[name] = [_replace_materials(item, assets) for item in value]
        elif (
            isinstance(value, tuple)
            and value
            and isinstance(value[0], BaseModel)
        ):
            updates[name] = tuple(
                _replace_materials(item, assets) for item in value
            )
    return record.model_copy(update=updates)


def _restore_image_paths(value: Any) -> Any:
    if isinstance(value, dict):
        restored = {
            key: _restore_image_paths(item) for key, item in value.items()
        }
        if (
            'key' in restored
            and 'payload' in restored
            and isinstance(restored['payload'], str)
        ):
            restored['payload'] = Path(restored['payload'])
        return restored
    if isinstance(value, list):
        return [_restore_image_paths(item) for item in value]
    return value


def hydrate_session(record: dict[str, Any]) -> SessionHistory:
    """Hydrate a complete session history from ordered Cypher maps."""
    session = LearningSession(
        uuid=record['uuid'],
        goal=record['goal'],
        targets=record['targets'],
        status=record['status'],
        created_at=native_datetime(record['created_at']),
        updated_at=native_datetime(record['updated_at']),
        invalidated_at=native_datetime(record['invalidated_at'])
        if record['invalidated_at'] is not None
        else None,
    )
    steps: list[SessionStep] = []
    responses: list[TaskResponse] = []
    assessments: list[TaskAssessment] = []
    for row in record['steps']:
        if row['kind'] == 'task':
            step = hydrate_task(row)
            for response_row in row['responses']:
                response = hydrate_task_response(response_row)
                responses.append(response)
                assessments.extend(
                    hydrate_task_assessment(item)
                    for item in response_row['assessments']
                )
        elif row['kind'] == 'teaching':
            payload = read_payload(row['payload_json'])
            step = Teaching(
                **payload,
                uuid=row['uuid'],
                created_at=native_datetime(row['created_at']),
                updated_at=native_datetime(row['updated_at']),
                invalidated_at=native_datetime(row['invalidated_at'])
                if row['invalidated_at'] is not None
                else None,
                target=row['target'],
                grounding=GroundingReferences(
                    proposition_uuids=row['proposition_uuids'],
                    argument_uuids=row['argument_uuids'],
                    unit_uuids=row['unit_uuids'],
                ),
                assessment_uuids=row['assessment_uuids'],
            )
        else:
            payload = read_payload(row['payload_json'])
            decision = payload['decision']
            decision['assessment_uuids'] = row['assessment_uuids']
            if decision['kind'] != 'finish':
                decision['target'] = row['target']
            step = LearningDecisionRecord(
                **payload,
                uuid=row['uuid'],
                created_at=native_datetime(row['created_at']),
                updated_at=native_datetime(row['updated_at']),
                invalidated_at=native_datetime(row['invalidated_at'])
                if row['invalidated_at'] is not None
                else None,
            )
        steps.append(step)
    return SessionHistory(
        session=session,
        steps=steps,
        responses=responses,
        assessments=assessments,
    )


def hydrate_task_response(row: dict[str, Any]) -> TaskResponse:
    """Hydrate a task response with its relationship-owned task UUID."""
    payload = read_payload(row['payload_json'])
    return TaskResponse(
        uuid=row['uuid'],
        task_uuid=row['task_uuid'],
        answer=TypeAdapter(TaskAnswer).validate_python(payload['answer']),
        created_at=native_datetime(row['created_at']),
        updated_at=native_datetime(row['updated_at']),
        invalidated_at=native_datetime(row['invalidated_at'])
        if row['invalidated_at'] is not None
        else None,
    )


def hydrate_task_assessment(row: dict[str, Any]) -> TaskAssessment:
    """Hydrate a task grade with its relationship-owned response UUID."""
    return TaskAssessment(
        uuid=row['uuid'],
        response_uuid=row['response_uuid'],
        grade=TaskGrade.model_validate(
            read_payload(row['payload_json'])['grade']
        ),
        created_at=native_datetime(row['created_at']),
        updated_at=native_datetime(row['updated_at']),
        invalidated_at=native_datetime(row['invalidated_at'])
        if row['invalidated_at'] is not None
        else None,
    )


def hydrate_task(row: dict[str, Any]) -> Task:
    payload = read_payload(row['payload_json'])
    solution_row = row['solution']
    solution_payload = read_payload(solution_row['payload_json'])
    solution = Solution(
        **solution_payload,
        uuid=solution_row['uuid'],
        created_at=native_datetime(solution_row['created_at']),
        updated_at=native_datetime(solution_row['updated_at']),
        invalidated_at=native_datetime(solution_row['invalidated_at'])
        if solution_row['invalidated_at'] is not None
        else None,
        grounding=GroundingReferences(
            proposition_uuids=solution_row['proposition_uuids'],
            argument_uuids=solution_row['argument_uuids'],
            unit_uuids=solution_row['unit_uuids'],
        ),
    )
    return Task(
        **payload,
        uuid=row['uuid'],
        created_at=native_datetime(row['created_at']),
        updated_at=native_datetime(row['updated_at']),
        invalidated_at=native_datetime(row['invalidated_at'])
        if row['invalidated_at'] is not None
        else None,
        target=TypeAdapter(LearningTarget).validate_python(row['target']),
        grounding=GroundingReferences(
            proposition_uuids=row['proposition_uuids'],
            argument_uuids=row['argument_uuids'],
            unit_uuids=row['unit_uuids'],
        ),
        solution=solution,
    )


def native_datetime(value: Any):
    """Convert Neo4j temporal values to Python datetimes."""
    return value if isinstance(value, datetime) else value.to_native()
