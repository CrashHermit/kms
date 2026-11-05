"""Real-driver checks for session-private teaching workspaces."""

import asyncio
import base64
import hashlib
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from neo4j.exceptions import ResultNotSingleError
from PIL import Image

from kms.config.services import DatabaseSettings
from kms.core.media_budget import ImagePayloadBudget
from kms.core.model.block import SourceBlock
from kms.core.model.page import SourcePage
from kms.core.model.source import Source
from kms.core.model.teaching.inference import (
    ReadImagesAction,
    ReadPassagesAction,
)
from kms.core.model.teaching.observation import (
    ExistingLearnerObservationReference,
    LearnerObservation,
    LearnerObservationKind,
    LearnerObservationWrite,
)
from kms.core.model.teaching.operation import (
    Clarify,
    CompleteProbing,
    RespondAction,
    TeachingAssessmentDraft,
    TeachingResponseSegment,
)
from kms.core.model.teaching.workflow import (
    TeachingInteractionResult,
    TeachingTurnRequest,
)
from kms.core.model.teaching.workspace import (
    TeachingEvidenceWorkspace,
    TeachingLearnerObservationRegistryEntry,
)
from kms.core.model.visual_asset import VisualAsset
from kms.core.source_snapshot import finalize_snapshot_identities
from kms.core.source_workspace import register_block
from kms.core.tokenization.budget import TokenBudget
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.database import schema
from kms.database.client import DatabaseClient
from kms.database.source.source_block_repository import SourceBlockRepository
from kms.database.source.source_snapshot_repository import (
    SourceSnapshotRepository,
)
from kms.database.teaching.observation_repository import (
    LearnerObservationRepository,
    write_learner_observations,
)
from kms.database.teaching.repository import TeachingRepository
from kms.langgraph.teaching.graph import TeachingGraph
from kms.langgraph.teaching.state import TeachingTurnState
from kms.node.teaching.history import TeachingHistoryNode
from kms.node.teaching.interaction import TeachingInteractionNode
from kms.node.teaching.interaction_tools import TeachingInteractionToolsNode
from kms.node.teaching.persistence import TeachingPersistenceNode
from kms.node.teaching.teacher import TeachingTeacherNode
from kms.node.teaching.tools import TeachingEvidenceTools
from kms.service.learner_observations import LearnerObservationService
from kms.service.source_evidence import SourceEvidenceService
from kms.service.source_workspace import SourceWorkspaceService
from kms.service.teaching import TeachingService

_REQUIRED = (
    'KMS_DATABASE__URI',
    'KMS_DATABASE__USERNAME',
    'KMS_DATABASE__PASSWORD',
    'KMS_DATABASE__DATABASE',
)


class _NoOpTurnStage:
    async def run(self, state):
        return {}


def _turn_graph(teacher, source_evidence, learner_observations, history_search):
    stage = _NoOpTurnStage()
    return TeachingGraph(
        TeachingHistoryNode(history_search),
        TeachingInteractionNode(source_evidence, learner_observations),
        teacher,
        teacher,
        teacher,
        TeachingInteractionToolsNode(None),
        stage,
        stage,
        stage,
        None,
        None,
        stage.run,
        max_tool_calls=0,
        max_research_tool_calls=0,
    )


class _EmptyLearnerObservationSearch:
    async def search(self, user_uuid, query):
        return []


@pytest.mark.skipif(
    os.getenv('KMS_NEO4J_IT') != '1'
    or not all(os.getenv(name) for name in _REQUIRED),
    reason='KMS Neo4j integration environment is not enabled',
)
def test_session_restores_active_evidence_from_owned_graph_for_later_citation(
    tmp_path: Path,
):
    async def exercise():
        database = DatabaseClient(
            DatabaseSettings(
                uri=os.environ[_REQUIRED[0]],
                username=os.environ[_REQUIRED[1]],
                password=os.environ[_REQUIRED[2]],
                database=os.environ[_REQUIRED[3]],
            )
        )
        teaching = TeachingService(TeachingRepository(database.session))
        snapshots = SourceSnapshotRepository(database.session)
        node = TeachingPersistenceNode(teaching)
        user = str(uuid4())
        image_path = tmp_path / 'figure.png'
        Image.new('RGB', (2, 2), 'blue').save(image_path)
        image_bytes = image_path.read_bytes()
        captured = finalize_snapshot_identities(
            Source(
                uuid=str(uuid4()),
                key='session evidence',
                metadata={
                    'embedding_profile': {'model': 'test', 'dimension': 1},
                },
            ),
            [
                SourcePage(
                    index=0,
                    blocks=[
                        SourceBlock(
                            block_type='image',
                            content='A useful diagram.',
                            embedding=[0.0],
                            assets=[
                                VisualAsset(
                                    path=str(image_path),
                                    content_sha256=hashlib.sha256(
                                        image_bytes
                                    ).hexdigest(),
                                )
                            ],
                        ),
                        SourceBlock(
                            block_type='code',
                            content='def double(x): return x * 2',
                            embedding=[0.0],
                        ),
                    ],
                )
            ],
        )
        source_uuid = captured.source.uuid
        session_uuid = None
        try:
            await schema.ensure_structural_schema(database.session)
            async with database.session() as session:
                result = await session.run(
                    'CREATE (:User {uuid: $uuid})', uuid=user
                )
                await result.consume()
            source_snapshot = await snapshots.create_owned_snapshot(
                user, captured.source, captured.pages
            )
            source_evidence = SourceEvidenceService(
                SourceBlockRepository(database.session)
            )
            block = source_snapshot.pages[0].blocks[0]
            asset = block.assets[0]
            created = await teaching.start_session(user, 'Understand diagrams')
            session_uuid = created.session.uuid
            snapshot = await teaching.load_snapshot(user, session_uuid)
            learner_observations = LearnerObservationService(
                LearnerObservationRepository(database.session)
            )
            workspace = TeachingEvidenceWorkspace()
            block_pos = register_block(
                workspace, source_snapshot.source, block, [0]
            )
            tool = TeachingEvidenceTools(
                None,
                source_evidence,
                SourceWorkspaceService(source_evidence),
                None,
                None,
                learner_observation_search=_EmptyLearnerObservationSearch(),
                learner_observations=learner_observations,
                image_budget=ImagePayloadBudget(4, 1_000_000, 1_000_000),
                discovery_preview_characters=320,
                catalog_page_size=10,
            )
            observation = await tool.read_passages(
                user,
                None,
                ReadPassagesAction(
                    kind='read_passages',
                    block_positions=[block_pos],
                    expand_memberships=True,
                    before=0,
                    after=0,
                ),
                workspace,
            )
            assert workspace.active_image_positions == []
            assert (
                workspace.image_registry[observation.image_positions[0]].data
                is None
            )
            image_pos = observation.image_positions[0]
            await tool.read_images(
                user,
                None,
                ReadImagesAction(
                    kind='read_images', image_positions=[image_pos]
                ),
                workspace,
            )
            assert workspace.receipts == []
            response = RespondAction(
                segments=[
                    TeachingResponseSegment(
                        content='The inspected diagram represents the relation.',
                        cited_evidence_positions=[],
                    )
                ],
                content_kind='coordination',
                operation=Clarify(kind='clarify'),
            )
            request = TeachingTurnRequest(
                user_uuid=user,
                session_uuid=session_uuid,
                expected_revision=0,
                snapshot=snapshot,
                source_uuids=None,
            )
            interaction = TeachingInteractionResult(
                response=response,
                workspace=workspace,
                visible_turn_indexes=[],
                visible_observation_indexes=[],
                visible_citation_receipt_indexes=[],
            )
            first = await node.run(
                TeachingTurnState(request=request, interaction=interaction)
            )
            assert first['final_view'].session.revision == 1
            assert (
                await snapshots.load_owned_snapshot(user, source_uuid)
                is not None
            )
            saved = await teaching.load_snapshot(user, session_uuid)
            assert saved.workspace is not None
            assert saved.workspace.block_registry[0].block_uuid == block.uuid
            assert not hasattr(
                await teaching.load_session(user, session_uuid),
                'workspace_json',
            )
            with pytest.raises(ResultNotSingleError):
                await teaching.load_snapshot(str(uuid4()), session_uuid)

            stale = TeachingInteractionResult(
                response=response,
                workspace=TeachingEvidenceWorkspace(),
                visible_turn_indexes=[],
                visible_observation_indexes=[],
                visible_citation_receipt_indexes=[],
            )
            with pytest.raises(ResultNotSingleError):
                await node.run(
                    TeachingTurnState(request=request, interaction=stale)
                )
            assert (
                await teaching.load_snapshot(user, session_uuid)
            ).workspace == saved.workspace
            assert (
                await snapshots.load_owned_snapshot(user, source_uuid)
                is not None
            )
            await teaching.submit_response(
                user, session_uuid, 'Tell me about it.', expected_revision=1
            )
            resumed = await teaching.load_snapshot(user, session_uuid)

            class Counter:
                def count_texts(self, texts):
                    return [len(text) for text in texts]

            class Teacher:
                async def aforward(self, *, context, content_images):
                    assert [
                        item.material.text for item in context.material.blocks
                    ] == ['A useful diagram.']
                    assert context.material.blocks[0].page_indexes == [0]
                    assert context.material.blocks[
                        0
                    ].material.image_positions == [0]
                    assert len(context.images) == 1
                    assert context.images[0].position == 0
                    assert len(content_images) == 1
                    assert (
                        base64.b64decode(content_images[0].url.split(',', 1)[1])
                        == image_bytes
                    )
                    return RespondAction(
                        segments=[
                            TeachingResponseSegment(
                                content='The diagram represents the relation.',
                                cited_evidence_positions=[1],
                            )
                        ],
                        content_kind='instruction',
                        operation=Clarify(kind='clarify'),
                    )

            teacher = TeachingTeacherNode(
                Teacher(),
                TokenBudget(
                    MaterialTokenEstimator(Counter()), token_limit=100000
                ),
                0,
                image_budget=ImagePayloadBudget(
                    max_images=4, max_bytes=1000000, max_pixels=1000000
                ),
                discovery_preview_characters=320,
                research_enabled=False,
            )
            graph = _turn_graph(
                teacher,
                source_evidence,
                learner_observations,
                _EmptyLearnerObservationSearch(),
            )
            next_request = TeachingTurnRequest(
                user_uuid=user,
                session_uuid=session_uuid,
                expected_revision=2,
                snapshot=resumed,
                source_uuids=None,
            )
            result_state = await graph.run(
                TeachingTurnState(request=next_request)
            )
            next_result = result_state.interaction
            assert next_result is not None
            assert next_result.workspace.active_block_positions == [block_pos]
            restored_snapshot = await snapshots.load_owned_snapshot(
                user, source_uuid
            )
            assert restored_snapshot is not None
            assert restored_snapshot.pages[0].blocks[1].content == (
                'def double(x): return x * 2'
            )
            assert next_result.workspace.active_image_positions == [image_pos]
            assert len(next_result.workspace.receipts) == 2
            assert [
                receipt.kind for receipt in next_result.workspace.receipts
            ] == ['block', 'image']
            assert (
                next_result.workspace.receipts[1].source.revision
                == source_snapshot.source.revision
            )
            assert next_result.workspace.receipts[1].data == image_bytes
            assert (
                next_result.workspace.image_registry[image_pos].data
                == image_bytes
            )
            await node.run(
                TeachingTurnState(request=next_request, interaction=next_result)
            )
            still_admitted = await snapshots.load_owned_snapshot(
                user, source_uuid
            )
            assert still_admitted is not None
            assert (
                still_admitted.source.revision
                == source_snapshot.source.revision
            )
            reloaded = await teaching.load_session(user, session_uuid)
            assert reloaded.turns[-1].citations[0].uuid == asset.uuid
            assert (
                reloaded.turns[-1].segments[0].citations[0].uuid == asset.uuid
            )
        finally:
            async with database.session() as session:
                result = await session.run(
                    'MATCH (u:User {uuid:$user}) OPTIONAL MATCH (u)-[:HAS_TEACHING_SESSION]->(s) OPTIONAL MATCH (s)-[:HAS_TURN]->(t) DETACH DELETE t,s,u',
                    user=user,
                )
                await result.consume()
                result = await session.run(
                    'MATCH (s:Source {uuid:$uuid}) OPTIONAL MATCH (s)-[:HAS_PAGE]->(p) OPTIONAL MATCH (p)-[:CONTAINS_BLOCK]->(b) OPTIONAL MATCH (b)-[:CONTAINS_VISUAL_ASSET]->(a) DETACH DELETE a,b,p,s',
                    uuid=source_uuid,
                )
                await result.consume()
            await database.close()

    asyncio.run(exercise())


@pytest.mark.skipif(
    os.getenv('KMS_NEO4J_IT') != '1'
    or not all(os.getenv(name) for name in _REQUIRED),
    reason='KMS Neo4j integration environment is not enabled',
)
def test_saved_learner_observation_restores_and_assessment_keeps_uuid():
    async def exercise():
        database = DatabaseClient(
            DatabaseSettings(
                uri=os.environ[_REQUIRED[0]],
                username=os.environ[_REQUIRED[1]],
                password=os.environ[_REQUIRED[2]],
                database=os.environ[_REQUIRED[3]],
            )
        )
        teaching = TeachingService(TeachingRepository(database.session))
        persistence = TeachingPersistenceNode(teaching)
        learner_observations = LearnerObservationService(
            LearnerObservationRepository(database.session)
        )
        user = str(uuid4())
        try:
            await schema.ensure_structural_schema(database.session)
            async with database.session() as session:
                result = await session.run(
                    'CREATE (:User {uuid: $uuid})', uuid=user
                )
                await result.consume()

            created = await teaching.start_session(user, 'Understand vectors')
            session_uuid = created.session.uuid
            snapshot = await teaching.load_snapshot(user, session_uuid)
            request = TeachingTurnRequest(
                user_uuid=user,
                session_uuid=session_uuid,
                expected_revision=0,
                snapshot=snapshot,
                source_uuids=None,
            )
            response = RespondAction(
                segments=[
                    TeachingResponseSegment(
                        content='What have you noticed so far?',
                        cited_evidence_positions=[],
                    )
                ],
                content_kind='coordination',
                operation=Clarify(kind='clarify'),
            )
            await persistence.run(
                TeachingTurnState(
                    request=request,
                    interaction=TeachingInteractionResult(
                        response=response,
                        workspace=TeachingEvidenceWorkspace(),
                        visible_turn_indexes=[],
                        visible_observation_indexes=[],
                        visible_citation_receipt_indexes=[],
                    ),
                )
            )
            await teaching.submit_response(
                user,
                session_uuid,
                'I can explain the direction change.',
                expected_revision=1,
            )
            snapshot = await teaching.load_snapshot(user, session_uuid)
            teacher_turn_uuid = snapshot.turns[0].turn.uuid
            learner_turn_uuid = snapshot.turns[-1].turn.uuid
            observation_uuids = [str(uuid4()) for _ in range(3)]
            now = datetime.now(UTC)
            writes = [
                LearnerObservationWrite(
                    observation=LearnerObservation(
                        uuid=uuid,
                        kind=LearnerObservationKind.GAP,
                        topic=f'Vector idea {position}',
                        description=f'Prior observation {position}.',
                        embedding=[0.0, 1.0],
                        created_at=now,
                    ),
                    evidence_turn_uuids=[learner_turn_uuid],
                )
                for position, uuid in enumerate(observation_uuids)
            ]
            async with database.session() as session:
                await session.execute_write(
                    write_learner_observations,
                    user,
                    teacher_turn_uuid,
                    writes,
                )
            records = await learner_observations.read(user, observation_uuids)
            workspace = TeachingEvidenceWorkspace(
                learner_observation_registry=[
                    TeachingLearnerObservationRegistryEntry(
                        observation=record.observation,
                        record=record if position == 2 else None,
                    )
                    for position, record in enumerate(records)
                ],
                active_learner_observation_positions=[2],
                discovered_learner_observation_positions=[2],
            )
            snapshot = await teaching.load_snapshot(user, session_uuid)
            workspace_request = TeachingTurnRequest(
                user_uuid=user,
                session_uuid=session_uuid,
                expected_revision=snapshot.session.revision,
                snapshot=snapshot,
                source_uuids=None,
            )
            await persistence.run(
                TeachingTurnState(
                    request=workspace_request,
                    interaction=TeachingInteractionResult(
                        response=response,
                        workspace=workspace,
                        visible_turn_indexes=[],
                        visible_observation_indexes=[],
                        visible_citation_receipt_indexes=[],
                    ),
                )
            )

            saved = await teaching.load_snapshot(user, session_uuid)
            assert saved.workspace is not None
            assert [
                entry.model_dump()
                for entry in saved.workspace.learner_observation_registry
            ] == [
                {'uuid': observation_uuids[0], 'was_read': False},
                {'uuid': observation_uuids[1], 'was_read': False},
                {'uuid': observation_uuids[2], 'was_read': True},
            ]
            assert saved.workspace.active_learner_observation_positions == [2]
            assert saved.workspace.discovered_learner_observation_positions == [
                2
            ]
            await teaching.submit_response(
                user,
                session_uuid,
                'Please check whether I have this right.',
                expected_revision=saved.session.revision,
            )
            saved = await teaching.load_snapshot(user, session_uuid)

            class Counter:
                def count_texts(self, texts):
                    return [len(text) for text in texts]

            class Teacher:
                async def aforward(self, *, context, content_images):
                    assert not content_images
                    assert len(context.learner_history) == 1
                    observation = context.learner_history[0]
                    assert observation.position == 0
                    assert observation.registry_position == 2
                    assert observation.description == 'Prior observation 2.'
                    assert observation.evidence[0].content == (
                        'I can explain the direction change.'
                    )
                    return RespondAction(
                        segments=[
                            TeachingResponseSegment(
                                content='You have demonstrated this idea.',
                                cited_evidence_positions=[],
                            )
                        ],
                        content_kind='instruction',
                        operation=CompleteProbing(
                            kind='complete_probing',
                            assessment=TeachingAssessmentDraft(
                                summary='The direction change is understood.',
                                strands=[],
                                observation_references=[
                                    ExistingLearnerObservationReference(
                                        kind='existing', position=0
                                    )
                                ],
                            ),
                        ),
                    )

            teacher = TeachingTeacherNode(
                Teacher(),
                TokenBudget(
                    MaterialTokenEstimator(Counter()), token_limit=100000
                ),
                0,
                image_budget=ImagePayloadBudget(
                    max_images=4, max_bytes=1_000_000, max_pixels=1_000_000
                ),
                discovery_preview_characters=320,
                research_enabled=False,
            )
            next_request = TeachingTurnRequest(
                user_uuid=user,
                session_uuid=session_uuid,
                expected_revision=saved.session.revision,
                snapshot=saved,
                source_uuids=None,
            )
            source_evidence = SourceEvidenceService(
                SourceBlockRepository(database.session)
            )
            graph = _turn_graph(
                teacher,
                source_evidence,
                learner_observations,
                _EmptyLearnerObservationSearch(),
            )
            result_state = await graph.run(
                TeachingTurnState(request=next_request)
            )
            interaction = result_state.interaction
            assert interaction is not None
            assert interaction.visible_observation_indexes == [2]
            await persistence.run(
                TeachingTurnState(
                    request=next_request,
                    interaction=interaction,
                )
            )

            async with database.session() as session:
                result = await session.run(
                    'MATCH (:User {uuid: $user})-[:HAS_TEACHING_SESSION]->'
                    '(:TeachingSession {uuid: $session})-[:HAS_ASSESSMENT]->'
                    '(:ProbeAssessment)-[:BASED_ON]->'
                    '(observation:LearnerObservation) '
                    'RETURN observation.uuid AS uuid ORDER BY uuid',
                    user=user,
                    session=session_uuid,
                )
                assessment_observation_uuids = [
                    record['uuid'] async for record in result
                ]
            assert assessment_observation_uuids == [observation_uuids[2]]
        finally:
            async with database.session() as session:
                result = await session.run(
                    'MATCH (user:User {uuid: $user})-[:HAS_LEARNER_OBSERVATION]->'
                    '(observation:LearnerObservation) DETACH DELETE observation',
                    user=user,
                )
                await result.consume()
                result = await session.run(
                    'MATCH (:User {uuid: $user})-[:HAS_TEACHING_SESSION]->'
                    '(:TeachingSession)-[:HAS_ASSESSMENT]->'
                    '(assessment:ProbeAssessment) DETACH DELETE assessment',
                    user=user,
                )
                await result.consume()
                result = await session.run(
                    'MATCH (user:User {uuid: $user}) '
                    'OPTIONAL MATCH (user)-[:HAS_TEACHING_SESSION]->(session) '
                    'OPTIONAL MATCH (session)-[:HAS_TURN]->(turn) '
                    'DETACH DELETE turn, session, user',
                    user=user,
                )
                await result.consume()
            await database.close()

    asyncio.run(exercise())
