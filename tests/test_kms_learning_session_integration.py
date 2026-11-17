"""Real Neo4j coverage for durable, owner-scoped learning sessions."""

import asyncio
import os
from pathlib import Path
from uuid import uuid4

import pytest
from neo4j.exceptions import ResultNotSingleError

from kms.artifacts import ContentAddressedArtifactStore
from kms.config.settings import Settings
from kms.database import schema
from kms.database.client import DatabaseClient
from kms.learning.references import (
    ArgumentTarget,
    GroundingReferences,
    PropositionTarget,
    SolutionTarget,
)
from kms.learning.session.context import select_session_context
from kms.learning.session.models import (
    AssessmentTarget,
    LearningDecisionRecord,
    LearningFinished,
    LearningSession,
    TeachingTarget,
)
from kms.learning.session.repository import LearningSessionRepository
from kms.learning.task.grading.models import TaskGradingRequest
from kms.learning.task.grading.node import TaskGradingNode
from kms.learning.task.models import (
    FreeAnswer,
    PresentedTask,
    ResponseInterpretation,
    ReviewRating,
    Solution,
    Task,
    TaskGrade,
    TaskResponse,
    UnknownTaskAnswer,
)
from kms.learning.teaching.models import Teaching
from kms.media.models import ImageAsset, ImagePart, Material, TextPart

_REQUIRED_ENVIRONMENT = (
    'KMS_DATABASE__URI',
    'KMS_DATABASE__USERNAME',
    'KMS_DATABASE__PASSWORD',
    'KMS_DATABASE__DATABASE',
)


def _interpretation() -> ResponseInterpretation:
    return ResponseInterpretation(
        reasoning_verdict='sound',
        explanation='The explanation identifies the filled point.',
        demonstrated_understandings=[
            'Reads the function value from the filled point.'
        ],
        gaps=[],
        possible_misconceptions=[],
        unresolved_questions=[],
    )


class _GradingModule:
    async def acall(self, *, request: TaskGradingRequest) -> TaskGrade:
        assert request.task.solution.uuid
        assert request.response.answer.kind == 'answer'
        return TaskGrade(
            correctness='correct',
            interpretation=_interpretation(),
            review_rating=ReviewRating.GOOD,
        )


def _target(
    understanding: str,
    assessment_uuids: list[str],
    target: PropositionTarget | ArgumentTarget | SolutionTarget | None = None,
) -> LearningDecisionRecord:
    return LearningDecisionRecord(
        decision=AssessmentTarget(
            understanding=understanding,
            explanation='Assess this part of the learning goal.',
            assessment_uuids=assessment_uuids,
            target=target,
        )
    )


@pytest.mark.skipif(
    os.getenv('KMS_NEO4J_IT') != '1'
    or not all(os.getenv(name) for name in _REQUIRED_ENVIRONMENT),
    reason='KMS Neo4j integration environment is not enabled',
)
def test_learning_sessions_are_durable_owner_scoped_and_serialized(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        database = DatabaseClient(Settings().database)
        suffix = uuid4().hex
        owner_uuid = f'learning-it-owner-{suffix}'
        other_owner_uuid = f'learning-it-other-owner-{suffix}'
        source_uuid = f'learning-it-source-{suffix}'
        other_source_uuid = f'learning-it-other-source-{suffix}'
        unit_uuid = f'learning-it-unit-{suffix}'
        other_unit_uuid = f'learning-it-other-unit-{suffix}'
        proposition_uuid = f'learning-it-proposition-{suffix}'
        other_proposition_uuid = f'learning-it-other-proposition-{suffix}'
        argument_uuid = f'learning-it-argument-{suffix}'
        other_argument_uuid = f'learning-it-other-argument-{suffix}'
        fixture_uuids = {
            owner_uuid,
            other_owner_uuid,
            source_uuid,
            other_source_uuid,
            unit_uuid,
            other_unit_uuid,
            proposition_uuid,
            other_proposition_uuid,
            argument_uuid,
            other_argument_uuid,
        }
        artifact_directory = tmp_path / 'artifacts'
        repository = LearningSessionRepository(
            database.session, ContentAddressedArtifactStore(artifact_directory)
        )
        session_uuid = f'learning-it-session-{suffix}'
        second_session_uuid = f'learning-it-second-session-{suffix}'
        session = LearningSession(
            uuid=session_uuid,
            goal='Distinguish function values from limits.',
            targets=['Read f(2) from a graph.'],
        )
        second_session = LearningSession(
            uuid=second_session_uuid,
            goal='A separate assessment session.',
            targets=['Read graph values.'],
        )
        fixture_uuids.update({session_uuid, second_session_uuid})
        image_bytes = b'\x89PNG\x00\xffnon-UTF-8-image-data'
        path_image_bytes = b'\x89PNG\x00\xfe-path-image-data'
        original_image_path = tmp_path / 'input-image.png'
        original_image_path.write_bytes(path_image_bytes)
        byte_image = ImageAsset(('integration', 'bytes'), image_bytes)
        path_image = ImageAsset(('integration', 'path'), original_image_path)
        try:
            await schema.ensure_structural_schema(database.session)
            async with database.session() as db_session:
                result = await db_session.run(
                    """
                    CREATE (owner:User {uuid: $owner_uuid})
                    CREATE (other:User {uuid: $other_owner_uuid})
                    CREATE (source:Source {
                        uuid: $source_uuid, key: $source_uuid,
                        created_at: datetime(), updated_at: datetime(),
                        metadata: '{}', revision: 0
                    })
                    CREATE (other_source:Source {
                        uuid: $other_source_uuid, key: $other_source_uuid,
                        created_at: datetime(), updated_at: datetime(),
                        metadata: '{}', revision: 0
                    })
                    CREATE (unit:SourceUnit {
                        uuid: $unit_uuid, created_at: datetime(),
                        updated_at: datetime(), kind: 'content'
                    })
                    CREATE (other_unit:SourceUnit {
                        uuid: $other_unit_uuid, created_at: datetime(),
                        updated_at: datetime(), kind: 'content'
                    })
                    CREATE (proposition:Proposition {
                        uuid: $proposition_uuid, name: 'Function value',
                        statement: 'The filled point gives f(2)=5.',
                        created_at: datetime(), updated_at: datetime()
                    })
                    CREATE (other_proposition:Proposition {
                        uuid: $other_proposition_uuid, name: 'Other value',
                        statement: 'This belongs to another user.',
                        created_at: datetime(), updated_at: datetime()
                    })
                    CREATE (argument:Argument {
                        uuid: $argument_uuid, premise_uuids: [$proposition_uuid],
                        conclusion_uuid: $proposition_uuid,
                        created_at: datetime(), updated_at: datetime()
                    })
                    CREATE (other_argument:Argument {
                        uuid: $other_argument_uuid,
                        premise_uuids: [$other_proposition_uuid],
                        conclusion_uuid: $other_proposition_uuid,
                        created_at: datetime(), updated_at: datetime()
                    })
                    CREATE (owner)-[:OWNS_SOURCE]->(source)
                    CREATE (other)-[:OWNS_SOURCE]->(other_source)
                    CREATE (source)-[:HAS_UNIT]->(unit)
                    CREATE (other_source)-[:HAS_UNIT]->(other_unit)
                    CREATE (owner)-[:OWNS_PROPOSITION]->(proposition)
                    CREATE (other)-[:OWNS_PROPOSITION]->(other_proposition)
                    CREATE (owner)-[:OWNS_ARGUMENT]->(argument)
                    CREATE (other)-[:OWNS_ARGUMENT]->(other_argument)
                    """,
                    owner_uuid=owner_uuid,
                    other_owner_uuid=other_owner_uuid,
                    source_uuid=source_uuid,
                    other_source_uuid=other_source_uuid,
                    unit_uuid=unit_uuid,
                    other_unit_uuid=other_unit_uuid,
                    proposition_uuid=proposition_uuid,
                    other_proposition_uuid=other_proposition_uuid,
                    argument_uuid=argument_uuid,
                    other_argument_uuid=other_argument_uuid,
                )
                await result.consume()

            await repository.create_session(owner_uuid, session)
            empty = await repository.read_session(owner_uuid, session_uuid)
            assert empty.session == session
            assert empty.steps == []
            assert empty.responses == []
            assert empty.assessments == []
            await repository.create_session(owner_uuid, second_session)

            task = Task(
                understanding='Distinguish function values from limits.',
                prompt=Material(
                    parts=(
                        TextPart(text='Use the evidence. '),
                        ImagePart(asset=byte_image),
                        TextPart(text=' Compare both points. '),
                        ImagePart(asset=path_image),
                    )
                ),
                target=PropositionTarget(uuid=proposition_uuid),
                grounding=GroundingReferences(
                    proposition_uuids=[proposition_uuid],
                    unit_uuids=[unit_uuid],
                ),
                solution=Solution(
                    material=Material(
                        parts=(
                            TextPart(text='f(2) equals five. '),
                            ImagePart(asset=byte_image),
                            TextPart(text='The limit is fifteen. '),
                            ImagePart(asset=path_image),
                            ImagePart(asset=byte_image),
                        )
                    ),
                    assessment_criteria=[
                        'Identifies the filled point as f(2).'
                    ],
                    grounding=GroundingReferences(
                        argument_uuids=[argument_uuid],
                        unit_uuids=[unit_uuid],
                    ),
                ),
            )
            fixture_uuids.update({task.uuid, task.solution.uuid})
            persisted_task = await repository.append_task(
                owner_uuid, second_session_uuid, task
            )
            atomic_session_uuid = f'learning-it-atomic-session-{suffix}'
            atomic_session = LearningSession(
                uuid=atomic_session_uuid,
                goal='Atomic target variants.',
                targets=['Assess one typed target.'],
            )
            fixture_uuids.add(atomic_session_uuid)
            await repository.create_session(owner_uuid, atomic_session)
            for target in (
                PropositionTarget(uuid=proposition_uuid),
                ArgumentTarget(uuid=argument_uuid),
                SolutionTarget(uuid=persisted_task.solution.uuid),
            ):
                atomic_task = Task(
                    understanding='Assess one selected target.',
                    prompt=Material.from_text('Explain the selected target.'),
                    target=target,
                    grounding=GroundingReferences(),
                    solution=Solution(
                        material=Material.from_text('Reference answer.'),
                        assessment_criteria=['Explains the target.'],
                        grounding=GroundingReferences(),
                    ),
                )
                fixture_uuids.update(
                    {atomic_task.uuid, atomic_task.solution.uuid}
                )
                await repository.append_task(
                    owner_uuid, atomic_session_uuid, atomic_task
                )
                async with database.session() as db_session:
                    result = await db_session.run(
                        """
                        MATCH (:User {uuid: $user_uuid})-[:OWNS_SESSION]->
                              (:LearningSession {uuid: $session_uuid})-[:HAS_STEP]->
                              (task:Task {uuid: $task_uuid})
                        OPTIONAL MATCH (task)-[:ASSESSES]->(target)
                        RETURN count(target) AS target_count,
                               collect(DISTINCT labels(target)) AS target_labels
                        """,
                        user_uuid=owner_uuid,
                        session_uuid=atomic_session_uuid,
                        task_uuid=atomic_task.uuid,
                    )
                    target_row = await result.single(strict=True)
                assert target_row['target_count'] == 1
                assert (
                    target.kind.capitalize() in target_row['target_labels'][0]
                )
                assert (
                    await repository.read_task(
                        owner_uuid, atomic_session_uuid, atomic_task.uuid
                    )
                ).target == target

            presented = PresentedTask(
                task_uuid=persisted_task.uuid,
                prompt=persisted_task.prompt,
            )
            presented_json = presented.model_dump_json()
            assert 'solution' not in presented_json
            assert 'assessment_criteria' not in presented_json
            assert 'target' not in presented_json
            assert 'grounding' not in presented_json
            assert 'f(2) equals five' not in presented_json
            original_path = path_image.payload
            assert isinstance(original_path, Path)
            reopened_task_repository = LearningSessionRepository(
                database.session,
                ContentAddressedArtifactStore(artifact_directory),
            )

            task_response = TaskResponse(
                task_uuid=task.uuid,
                answer=FreeAnswer(
                    material=Material(
                        parts=(
                            TextPart(text='The value is five. '),
                            ImagePart(asset=byte_image),
                            TextPart(text='I used the filled point. '),
                            ImagePart(asset=path_image),
                            ImagePart(asset=byte_image),
                        )
                    ),
                    confidence='high',
                ),
            )
            fixture_uuids.add(task_response.uuid)
            persisted_response = await repository.record_task_response(
                owner_uuid, second_session_uuid, task_response
            )
            assert persisted_response.task_uuid == task.uuid
            assert (
                task_response.answer.material.images[1].payload == original_path
            )
            grade = (
                await TaskGradingNode(_GradingModule(), repository).run(
                    {
                        'user_uuid': owner_uuid,
                        'session_uuid': second_session_uuid,
                        'response_uuid': task_response.uuid,
                    }
                )
            )['assessment']
            fixture_uuids.add(grade.uuid)
            assert grade.grade.correctness == 'correct'
            async with database.session() as db_session:
                result = await db_session.run(
                    """
                    MATCH (response:TaskResponse {uuid: $response_uuid})
                    MATCH (assessment:TaskAssessment {uuid: $assessment_uuid})
                    RETURN properties(response) AS response_properties,
                           properties(assessment) AS assessment_properties
                    """,
                    response_uuid=task_response.uuid,
                    assessment_uuid=grade.uuid,
                )
                persisted_properties = await result.single(strict=True)
            assert (
                'task_uuid' not in persisted_properties['response_properties']
            )
            assert (
                'task_uuid'
                not in persisted_properties['response_properties'][
                    'payload_json'
                ]
            )
            assert (
                'response_uuid'
                not in persisted_properties['assessment_properties']
            )
            assert (
                'response_uuid'
                not in persisted_properties['assessment_properties'][
                    'payload_json'
                ]
            )
            hydrated_response = (
                await reopened_task_repository.read_task_response(
                    owner_uuid, second_session_uuid, task_response.uuid
                )
            )
            assert hydrated_response == persisted_response
            for inaccessible_owner, inaccessible_session in (
                (other_owner_uuid, second_session_uuid),
                (owner_uuid, session_uuid),
            ):
                with pytest.raises(ResultNotSingleError):
                    await reopened_task_repository.read_task_response(
                        inaccessible_owner,
                        inaccessible_session,
                        task_response.uuid,
                    )
            for part, expected in zip(
                (
                    part
                    for part in hydrated_response.answer.material.parts
                    if isinstance(part, ImagePart)
                ),
                (image_bytes, path_image_bytes, image_bytes),
                strict=True,
            ):
                assert isinstance(part.asset.payload, Path)
                assert (
                    ContentAddressedArtifactStore.read(part.asset.payload)
                    == expected
                )
            hydrated_task = await reopened_task_repository.read_task(
                owner_uuid, second_session_uuid, task.uuid
            )
            assert task.prompt.images[1].payload == original_path
            task_history = await reopened_task_repository.read_session(
                owner_uuid, second_session_uuid
            )
            assert persisted_response in task_history.responses
            assert grade in task_history.assessments
            repeat_grade = grade.model_copy(
                update={
                    'uuid': f'learning-it-repeat-grade-{suffix}',
                    'grade': grade.grade.model_copy(
                        update={'review_rating': ReviewRating.HARD}
                    ),
                }
            )
            fixture_uuids.add(repeat_grade.uuid)
            await repository.record_task_assessment(
                owner_uuid, second_session_uuid, repeat_grade
            )
            task_decision = _target(
                'Review the response.',
                [grade.uuid],
                PropositionTarget(uuid=proposition_uuid),
            )
            fixture_uuids.add(task_decision.uuid)
            await repository.append_decision(
                owner_uuid, second_session_uuid, task_decision
            )
            teaching_decision = LearningDecisionRecord(
                decision=TeachingTarget(
                    understanding=task.understanding,
                    objective='Explain the filled point.',
                    explanation='The observed answer confused the endpoints.',
                    assessment_uuids=[grade.uuid],
                    target=PropositionTarget(uuid=proposition_uuid),
                )
            )
            fixture_uuids.add(teaching_decision.uuid)
            await repository.append_decision(
                owner_uuid, second_session_uuid, teaching_decision
            )
            teaching = Teaching(
                understanding=task.understanding,
                objective='Distinguish the filled point from the open point.',
                target=PropositionTarget(uuid=proposition_uuid),
                material=Material(
                    parts=(
                        TextPart(text='The filled point establishes f(2)=5. '),
                        ImagePart(asset=path_image),
                    )
                ),
                grounding=GroundingReferences(
                    proposition_uuids=[proposition_uuid],
                    unit_uuids=[unit_uuid],
                ),
                assessment_uuids=[grade.uuid],
            )
            fixture_uuids.add(teaching.uuid)
            persisted_teaching = await repository.append_teaching(
                owner_uuid, second_session_uuid, teaching
            )
            latest_task_history = await reopened_task_repository.read_session(
                owner_uuid, second_session_uuid
            )
            assert grade in latest_task_history.assessments
            assert repeat_grade in latest_task_history.assessments
            assert latest_task_history.steps[-3].decision.assessment_uuids == [
                grade.uuid
            ]
            assert latest_task_history.steps[-3].decision.target == (
                PropositionTarget(uuid=proposition_uuid)
            )
            assert latest_task_history.steps[-2].decision.assessment_uuids == [
                grade.uuid
            ]
            assert latest_task_history.steps[-2].decision.target == (
                PropositionTarget(uuid=proposition_uuid)
            )
            assert latest_task_history.steps[-1] == persisted_teaching
            assert isinstance(latest_task_history.steps[-1], Teaching)
            assert latest_task_history.steps[-1].target == teaching.target
            assert latest_task_history.steps[-1].grounding == teaching.grounding
            assert latest_task_history.steps[-1].assessment_uuids == [
                grade.uuid
            ]
            assert isinstance(
                latest_task_history.steps[-1].material.parts[1].asset.payload,
                Path,
            )
            assert (
                ContentAddressedArtifactStore.read(
                    latest_task_history.steps[-1]
                    .material.parts[1]
                    .asset.payload
                )
                == path_image_bytes
            )
            async with database.session() as db_session:
                result = await db_session.run(
                    """
                    MATCH (teaching:Teaching {uuid: $teaching_uuid})
                    MATCH (teaching)-[:TEACHES]->(target)
                    OPTIONAL MATCH (teaching)-[:GROUNDED_IN]->(grounding)
                    OPTIONAL MATCH (teaching)-[:BASED_ON]->(assessment)
                    RETURN labels(target) AS target_labels,
                           target.uuid AS target_uuid,
                           collect(DISTINCT grounding.uuid) AS grounding_uuids,
                           collect(DISTINCT assessment.uuid) AS assessment_uuids
                    """,
                    teaching_uuid=teaching.uuid,
                )
                teaching_links = await result.single(strict=True)
            assert teaching_links['target_uuid'] == proposition_uuid
            assert 'Proposition' in teaching_links['target_labels']
            assert set(teaching_links['grounding_uuids']) == {
                proposition_uuid,
                unit_uuid,
            }
            assert teaching_links['assessment_uuids'] == [grade.uuid]
            async with database.session() as db_session:
                result = await db_session.run(
                    """
                    MATCH (decision:LearningDecision {uuid: $decision_uuid})
                    MATCH (decision)-[:TARGETS]->(target)
                    MATCH (decision)-[:BASED_ON]->(assessment)
                    RETURN labels(target) AS target_labels,
                           target.uuid AS target_uuid,
                           collect(assessment.uuid) AS assessment_uuids
                    """,
                    decision_uuid=task_decision.uuid,
                )
                decision_links = await result.single(strict=True)
            assert decision_links['target_uuid'] == proposition_uuid
            assert 'Proposition' in decision_links['target_labels']
            assert decision_links['assessment_uuids'] == [grade.uuid]
            async with database.session() as db_session:
                result = await db_session.run(
                    """
                    MATCH (decision:LearningDecision {uuid: $decision_uuid})
                    MATCH (decision)-[:TARGETS]->(target)
                    MATCH (decision)-[:BASED_ON]->(assessment)
                    RETURN labels(target) AS target_labels,
                           target.uuid AS target_uuid,
                           collect(assessment.uuid) AS assessment_uuids
                    """,
                    decision_uuid=teaching_decision.uuid,
                )
                teaching_decision_links = await result.single(strict=True)
            assert teaching_decision_links['target_uuid'] == proposition_uuid
            assert 'Proposition' in teaching_decision_links['target_labels']
            assert teaching_decision_links['assessment_uuids'] == [grade.uuid]
            task_context = select_session_context(
                latest_task_history,
                [task.uuid, task_decision.uuid, teaching.uuid],
            )
            assert persisted_response in task_context.responses
            assert grade in task_context.assessments
            assert len(task_history.steps) == 1
            assert isinstance(task_history.steps[0], Task)
            assert task_history.steps[0] == hydrated_task
            assert hydrated_task.created_at == task.created_at
            assert hydrated_task.updated_at == task.updated_at
            assert hydrated_task.invalidated_at == task.invalidated_at
            assert hydrated_task.solution.created_at == task.solution.created_at
            assert hydrated_task.solution.updated_at == task.solution.updated_at
            assert (
                hydrated_task.solution.invalidated_at
                == task.solution.invalidated_at
            )
            with pytest.raises(ResultNotSingleError):
                await reopened_task_repository.read_task(
                    other_owner_uuid, second_session_uuid, task.uuid
                )
            assert hydrated_task.uuid == task.uuid
            assert hydrated_task.solution.uuid == task.solution.uuid
            assert hydrated_task.target == task.target
            assert hydrated_task.grounding == task.grounding
            assert hydrated_task.solution.grounding == task.solution.grounding
            assert hydrated_task.target.uuid == proposition_uuid
            assert hydrated_task.solution.grounding.argument_uuids == [
                argument_uuid
            ]
            for material, expected_images in (
                (hydrated_task.prompt, (image_bytes, path_image_bytes)),
                (
                    hydrated_task.solution.material,
                    (image_bytes, path_image_bytes, image_bytes),
                ),
            ):
                assert [part.type for part in material.parts] == (
                    ['text', 'image', 'text', 'image']
                    if material is hydrated_task.prompt
                    else ['text', 'image', 'text', 'image', 'image']
                )
                for part, expected in zip(
                    (
                        part
                        for part in material.parts
                        if isinstance(part, ImagePart)
                    ),
                    expected_images,
                    strict=True,
                ):
                    assert isinstance(part.asset.payload, Path)
                    assert (
                        ContentAddressedArtifactStore.read(part.asset.payload)
                        == expected
                    )

            before_bad_task = await repository.read_session(
                owner_uuid, second_session_uuid
            )
            bad_target = persisted_task.model_copy(
                update={
                    'uuid': f'learning-it-bad-target-{suffix}',
                    'target': PropositionTarget(uuid=other_proposition_uuid),
                    'solution': persisted_task.solution.model_copy(
                        update={
                            'uuid': f'learning-it-bad-target-solution-{suffix}'
                        }
                    ),
                }
            )
            bad_solution = persisted_task.model_copy(
                update={
                    'uuid': f'learning-it-bad-solution-{suffix}',
                    'solution': persisted_task.solution.model_copy(
                        update={
                            'uuid': f'learning-it-bad-solution-record-{suffix}',
                            'grounding': GroundingReferences(
                                argument_uuids=[other_argument_uuid]
                            ),
                        }
                    ),
                }
            )
            fixture_uuids.update(
                {
                    bad_target.uuid,
                    bad_target.solution.uuid,
                    bad_solution.uuid,
                    bad_solution.solution.uuid,
                }
            )
            for invalid_task in (bad_target, bad_solution):
                with pytest.raises(ResultNotSingleError):
                    await repository.append_task(
                        owner_uuid, second_session_uuid, invalid_task
                    )
            after_bad_task = await repository.read_session(
                owner_uuid, second_session_uuid
            )
            assert [step.uuid for step in after_bad_task.steps] == [
                step.uuid for step in before_bad_task.steps
            ]
            async with database.session() as db_session:
                result = await db_session.run(
                    """
                    MATCH (node)
                    WHERE node.uuid IN $task_uuids + $solution_uuids
                    RETURN collect(node.uuid) AS uuids
                    """,
                    task_uuids=[bad_target.uuid, bad_solution.uuid],
                    solution_uuids=[
                        bad_target.solution.uuid,
                        bad_solution.solution.uuid,
                    ],
                )
                task_rows = await result.single(strict=True)
            assert task_rows['uuids'] == []
            original_path.unlink()
            reread_task = await reopened_task_repository.read_task(
                owner_uuid, second_session_uuid, task.uuid
            )
            assert (
                reread_task.prompt.images[1].payload.read_bytes()
                == path_image_bytes
            )
            reread_response = await reopened_task_repository.read_task_response(
                owner_uuid, second_session_uuid, task_response.uuid
            )
            assert [
                ContentAddressedArtifactStore.read(part.asset.payload)
                for part in reread_response.answer.material.parts
                if isinstance(part, ImagePart)
            ] == [image_bytes, path_image_bytes, image_bytes]

            with pytest.raises(ResultNotSingleError):
                await repository.read_session(other_owner_uuid, session_uuid)
            outsider_task = persisted_task.model_copy(
                update={
                    'uuid': f'learning-it-outsider-task-{suffix}',
                    'solution': persisted_task.solution.model_copy(
                        update={
                            'uuid': f'learning-it-outsider-solution-{suffix}'
                        }
                    ),
                }
            )
            fixture_uuids.update(
                {outsider_task.uuid, outsider_task.solution.uuid}
            )
            with pytest.raises(ResultNotSingleError):
                await repository.append_task(
                    other_owner_uuid, session_uuid, outsider_task
                )

            outsider_task_response = TaskResponse(
                task_uuid=task.uuid,
                answer=UnknownTaskAnswer(),
            )
            fixture_uuids.add(outsider_task_response.uuid)
            with pytest.raises(ResultNotSingleError):
                await repository.record_task_response(
                    other_owner_uuid,
                    second_session_uuid,
                    outsider_task_response,
                )
            outsider_task_grade = grade.model_copy(
                update={
                    'uuid': f'learning-it-outsider-grade-{suffix}',
                    'response_uuid': task_response.uuid,
                }
            )
            fixture_uuids.add(outsider_task_grade.uuid)
            with pytest.raises(ResultNotSingleError):
                await repository.record_task_assessment(
                    other_owner_uuid, second_session_uuid, outsider_task_grade
                )

            cross_session_decision = _target(
                'Do not cite another session.', [grade.uuid]
            )
            fixture_uuids.add(cross_session_decision.uuid)
            before_foreign_citation = await repository.read_session(
                owner_uuid, session_uuid
            )
            with pytest.raises(ResultNotSingleError):
                await repository.append_decision(
                    owner_uuid, session_uuid, cross_session_decision
                )
            after_foreign_citation = await repository.read_session(
                owner_uuid, session_uuid
            )
            assert [step.uuid for step in after_foreign_citation.steps] == [
                step.uuid for step in before_foreign_citation.steps
            ]

            foreign_target_decision = _target(
                'Do not target another learner’s proposition.',
                [grade.uuid],
                PropositionTarget(uuid=other_proposition_uuid),
            )
            fixture_uuids.add(foreign_target_decision.uuid)
            with pytest.raises(ResultNotSingleError):
                await repository.append_decision(
                    owner_uuid, session_uuid, foreign_target_decision
                )
            after_foreign_target = await repository.read_session(
                owner_uuid, session_uuid
            )
            assert [step.uuid for step in after_foreign_target.steps] == [
                step.uuid for step in before_foreign_citation.steps
            ]
            async with database.session() as db_session:
                result = await db_session.run(
                    'MATCH (decision:LearningDecision {uuid: $uuid}) '
                    'RETURN count(decision) AS count',
                    uuid=foreign_target_decision.uuid,
                )
                row = await result.single(strict=True)
            assert row['count'] == 0

            for index, updates in enumerate(
                (
                    {
                        'grounding': GroundingReferences(
                            unit_uuids=[other_unit_uuid]
                        )
                    },
                    {'assessment_uuids': [grade.uuid]},
                    {'target': PropositionTarget(uuid=other_proposition_uuid)},
                )
            ):
                invalid_teaching = persisted_teaching.model_copy(
                    update={
                        **updates,
                        'uuid': f'learning-it-invalid-teaching-{index}-{suffix}',
                    }
                )
                fixture_uuids.add(invalid_teaching.uuid)
                with pytest.raises(ResultNotSingleError):
                    await repository.append_teaching(
                        owner_uuid, session_uuid, invalid_teaching
                    )
                after_invalid_teaching = await repository.read_session(
                    owner_uuid, session_uuid
                )
                assert [step.uuid for step in after_invalid_teaching.steps] == [
                    step.uuid for step in before_foreign_citation.steps
                ]
                async with database.session() as db_session:
                    result = await db_session.run(
                        'MATCH (teaching:Teaching {uuid: $uuid}) '
                        'RETURN count(teaching) AS count',
                        uuid=invalid_teaching.uuid,
                    )
                    row = await result.single(strict=True)
                assert row['count'] == 0

            for index, grounding in enumerate(
                (
                    GroundingReferences(
                        proposition_uuids=[other_proposition_uuid],
                        argument_uuids=[other_argument_uuid],
                        unit_uuids=[other_unit_uuid],
                    ),
                    GroundingReferences(
                        proposition_uuids=[proposition_uuid],
                        argument_uuids=[argument_uuid],
                        unit_uuids=[other_unit_uuid],
                    ),
                )
            ):
                foreign_task = persisted_task.model_copy(
                    update={
                        'uuid': f'learning-it-foreign-task-{index}-{suffix}',
                        'grounding': grounding,
                        'solution': persisted_task.solution.model_copy(
                            update={
                                'uuid': f'learning-it-foreign-solution-{index}-{suffix}'
                            }
                        ),
                    }
                )
                fixture_uuids.update(
                    {foreign_task.uuid, foreign_task.solution.uuid}
                )
                before_foreign_grounding = await repository.read_session(
                    owner_uuid, session_uuid
                )
                with pytest.raises(ResultNotSingleError):
                    await repository.append_task(
                        owner_uuid, session_uuid, foreign_task
                    )
                after_foreign_grounding = await repository.read_session(
                    owner_uuid, session_uuid
                )
                assert [
                    step.uuid for step in after_foreign_grounding.steps
                ] == [step.uuid for step in before_foreign_grounding.steps]
                async with database.session() as db_session:
                    result = await db_session.run(
                        """
                        OPTIONAL MATCH (task:Task {uuid: $task_uuid})
                        OPTIONAL MATCH (task)-[grounding:GROUNDED_IN]->()
                        OPTIONAL MATCH (solution:Solution {uuid: $solution_uuid})
                        RETURN count(DISTINCT task) AS tasks,
                               count(grounding) AS grounding_links,
                               count(DISTINCT solution) AS solutions
                        """,
                        task_uuid=foreign_task.uuid,
                        solution_uuid=foreign_task.solution.uuid,
                    )
                    row = await result.single(strict=True)
                assert row == {'tasks': 0, 'grounding_links': 0, 'solutions': 0}

            concurrent_first = _target('Concurrent target one.', [])
            concurrent_second = _target('Concurrent target two.', [])
            fixture_uuids.update(
                {concurrent_first.uuid, concurrent_second.uuid}
            )
            repo_a = LearningSessionRepository(
                database.session,
                ContentAddressedArtifactStore(artifact_directory),
            )
            repo_b = LearningSessionRepository(
                database.session,
                ContentAddressedArtifactStore(artifact_directory),
            )
            await asyncio.gather(
                repo_a.append_decision(
                    owner_uuid, second_session_uuid, concurrent_first
                ),
                repo_b.append_decision(
                    owner_uuid, second_session_uuid, concurrent_second
                ),
            )

            finish = LearningDecisionRecord(
                decision=LearningFinished(
                    summary=(
                        'The learner read the filled point correctly and gave '
                        'supporting reasoning; explain that the limit remains a '
                        'separate unresolved idea.'
                    ),
                    assessment_uuids=[grade.uuid],
                )
            )
            fixture_uuids.add(finish.uuid)
            await repo_b.append_decision(
                owner_uuid, second_session_uuid, finish
            )

            reopened_repository = LearningSessionRepository(
                database.session,
                ContentAddressedArtifactStore(artifact_directory),
            )
            history = await reopened_repository.read_session(
                owner_uuid, second_session_uuid
            )
            assert history.session.status == 'completed'
            assert [step.uuid for step in history.steps[:4]] == [
                task.uuid,
                task_decision.uuid,
                teaching_decision.uuid,
                teaching.uuid,
            ]
            assert len(history.steps) == 7
            assert history.steps[-1].uuid == finish.uuid
            assert history.steps[-1].decision == finish.decision
            assert set(step.uuid for step in history.steps[-3:-1]) == {
                concurrent_first.uuid,
                concurrent_second.uuid,
            }
            assert history.responses == [persisted_response]
            assert {item.uuid for item in history.assessments} == {
                grade.uuid,
                repeat_grade.uuid,
            }
            assert grade in history.assessments
            assert repeat_grade in history.assessments
            hydrated_task = history.steps[0]
            assert isinstance(hydrated_task, Task)
            assert hydrated_task == reread_task
            assert history.steps[1].decision.assessment_uuids == [grade.uuid]
            assert history.steps[1].decision.target == PropositionTarget(
                uuid=proposition_uuid
            )
            assert history.steps[2].decision.assessment_uuids == [grade.uuid]
            assert history.steps[2].decision.target == PropositionTarget(
                uuid=proposition_uuid
            )
            assert history.steps[3] == persisted_teaching
            assert history.responses[0].task_uuid == task.uuid
            work = history.responses[0].answer.material
            assert [part.type for part in work.parts] == [
                'text',
                'image',
                'text',
                'image',
                'image',
            ]
            for material, expected_images in (
                (work, (image_bytes, path_image_bytes, image_bytes)),
                (hydrated_task.prompt, (image_bytes, path_image_bytes)),
            ):
                for part, expected in zip(
                    (
                        part
                        for part in material.parts
                        if isinstance(part, ImagePart)
                    ),
                    expected_images,
                    strict=True,
                ):
                    assert isinstance(part.asset.payload, Path)
                    assert (
                        ContentAddressedArtifactStore.read(part.asset.payload)
                        == expected
                    )
            assert finish.decision.assessment_uuids == [grade.uuid]

            async with database.session() as db_session:
                result = await db_session.run(
                    """
                    MATCH (session:LearningSession {uuid: $session_uuid})
                    CALL (session) {
                        OPTIONAL MATCH (session)-[:FIRST_STEP]->(step)
                        RETURN collect(step.uuid) AS first_steps
                    }
                    CALL (session) {
                        OPTIONAL MATCH (session)-[:LAST_STEP]->(step)
                        RETURN collect(step.uuid) AS last_steps
                    }
                    CALL (session) {
                        OPTIONAL MATCH (left:LearningSessionStep)-[:NEXT_STEP]->(right)
                        WHERE (session)-[:HAS_STEP]->(left)
                        RETURN collect([left.uuid, right.uuid]) AS edges
                    }
                    RETURN first_steps, last_steps, edges
                    """,
                    session_uuid=second_session_uuid,
                )
                chain = await result.single(strict=True)
            assert chain['first_steps'] == [task.uuid]
            assert chain['last_steps'] == [finish.uuid]
            assert set(map(tuple, chain['edges'])) == {
                (left.uuid, right.uuid)
                for left, right in zip(
                    history.steps, history.steps[1:], strict=False
                )
            }
            assert len(chain['edges']) == len(history.steps) - 1

            finished_task = persisted_task.model_copy(
                update={
                    'uuid': f'learning-it-finished-task-{suffix}',
                    'solution': persisted_task.solution.model_copy(
                        update={
                            'uuid': f'learning-it-finished-solution-{suffix}'
                        }
                    ),
                }
            )
            fixture_uuids.update(
                {finished_task.uuid, finished_task.solution.uuid}
            )
            with pytest.raises(ResultNotSingleError):
                await reopened_repository.append_task(
                    owner_uuid, second_session_uuid, finished_task
                )
        finally:
            async with database.session() as db_session:
                result = await db_session.run(
                    """
                    MATCH (owner:User)
                    WHERE owner.uuid IN $owner_uuids
                    OPTIONAL MATCH (owner)-[:OWNS_SESSION]->(session)
                    OPTIONAL MATCH (session)-[:HAS_STEP]->(step)
                    OPTIONAL MATCH (step)-[:HAS_SOLUTION]->(solution)
                    OPTIONAL MATCH (step)-[:HAS_RESPONSE]->(response)
                    OPTIONAL MATCH (response)-[:HAS_ASSESSMENT]->(assessment)
                    WITH collect(DISTINCT session) +
                         collect(DISTINCT step) +
                         collect(DISTINCT solution) +
                         collect(DISTINCT response) +
                         collect(DISTINCT assessment) AS records
                    UNWIND records AS record
                    DETACH DELETE record
                    """,
                    owner_uuids=[owner_uuid, other_owner_uuid],
                )
                await result.consume()
                result = await db_session.run(
                    """
                    MATCH (node)
                    WHERE node.uuid IN $fixture_uuids
                    DETACH DELETE node
                    """,
                    fixture_uuids=list(fixture_uuids),
                )
                await result.consume()
            await database.close()

    asyncio.run(exercise())
