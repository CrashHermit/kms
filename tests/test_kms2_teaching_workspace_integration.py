"""Real-driver checks for session-private teaching workspaces."""

import asyncio
import os
from pathlib import Path
from uuid import uuid4

import pytest
from neo4j.exceptions import ResultNotSingleError
from PIL import Image

from kms2.config.services import DatabaseSettings
from kms2.core.media_budget import ImagePayloadBudget
from kms2.core.model.block import SourceBlock
from kms2.core.model.page import SourcePage
from kms2.core.model.source import Source
from kms2.core.model.teaching.operation import Clarify, RespondAction
from kms2.core.model.teaching.workflow import (
    TeachingInteractionRequest,
    TeachingInteractionResult,
    TeachingTurnRequest,
)
from kms2.core.model.teaching.workspace import (
    BlockInspection,
    ImageInspection,
    TeachingEvidenceWorkspace,
)
from kms2.core.model.visual_asset import VisualAsset
from kms2.core.source_snapshot import finalize_snapshot_identities
from kms2.core.windowing import TokenBudget
from kms2.database import schema
from kms2.database.client import DatabaseClient
from kms2.database.source.source_snapshot_repository import (
    SourceSnapshotRepository,
)
from kms2.database.teaching.repository import TeachingRepository
from kms2.langgraph.teaching.interaction_graph import TeachingInteractionGraph
from kms2.langgraph.teaching.state import TeachingTurnState
from kms2.node.teaching.evidence import register_block, register_image
from kms2.node.teaching.interaction_tools import TeachingInteractionToolsNode
from kms2.node.teaching.persistence import TeachingPersistenceNode
from kms2.node.teaching.teacher import TeachingTeacherNode
from kms2.service.teaching import TeachingService

_REQUIRED = (
    'KMS2_DATABASE__URI',
    'KMS2_DATABASE__USERNAME',
    'KMS2_DATABASE__PASSWORD',
    'KMS2_DATABASE__DATABASE',
)


@pytest.mark.skipif(
    os.getenv('KMS2_NEO4J_IT') != '1'
    or not all(os.getenv(name) for name in _REQUIRED),
    reason='KMS2 Neo4j integration environment is not enabled',
)
def test_session_restores_unpublished_active_evidence_for_later_citation(
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
        staged = finalize_snapshot_identities(
            Source(uuid=str(uuid4()), key='session evidence'),
            [
                SourcePage(
                    index=17,
                    blocks=[
                        SourceBlock(
                            block_type='image',
                            content='A useful diagram.',
                            assets=[VisualAsset(path=str(image_path))],
                        )
                    ],
                )
            ],
        )
        block = staged.pages[0].blocks[0]
        asset = block.assets[0]
        session_uuid = None
        try:
            await schema.ensure_structural_schema(database.session)
            async with database.session() as session:
                result = await session.run(
                    'CREATE (:User {uuid: $uuid})', uuid=user
                )
                await result.consume()
            created = await teaching.start_session(user, 'Understand diagrams')
            session_uuid = created.session.uuid
            snapshot = await teaching.load_snapshot(user, session_uuid)
            workspace = TeachingEvidenceWorkspace(
                staged_sources=[staged], acquired_sources=[staged]
            )
            block_pos = register_block(workspace, staged.source, block, [17])
            image_pos = register_image(
                workspace, asset, staged.source, block_pos
            )
            workspace.image_registry[image_pos].data = image_bytes
            workspace.inspected_block_positions.add(block_pos)
            workspace.inspected_image_positions.add(image_pos)
            workspace.active_block_positions = [block_pos]
            workspace.active_image_positions = [image_pos]
            workspace.receipts = [
                BlockInspection(
                    source=staged.source, block=block, page_indexes=[17]
                ),
                ImageInspection(
                    source=staged.source,
                    asset=asset,
                    block_uuids=[block.uuid],
                    data=image_bytes,
                ),
            ]
            response = RespondAction(
                content='Consider the figure.',
                content_kind='coordination',
                operation=Clarify(kind='clarify'),
                cited_evidence_positions=[],
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
                visible_citation_receipt_indexes=[0, 1],
            )
            first = await node.run(
                TeachingTurnState(request=request, interaction=interaction)
            )
            assert first['final_view'].session.revision == 1
            assert (
                await snapshots.load_owned_snapshot(user, staged.source.uuid)
                is None
            )
            saved = await teaching.load_snapshot(user, session_uuid)
            assert saved.workspace_json is not None
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
            ).workspace_json == saved.workspace_json
            await teaching.submit_response(
                user, session_uuid, 'Tell me about it.', expected_revision=1
            )
            resumed = await teaching.load_snapshot(user, session_uuid)

            class Counter:
                def count_texts(self, texts):
                    return [len(text) for text in texts]

            class Teacher:
                async def aforward(self, *, context, evidence_images):
                    assert [item.content for item in context.blocks] == [
                        'A useful diagram.'
                    ]
                    assert context.blocks[0].page_indexes == [17]
                    assert (
                        len(context.images) == 1 and len(evidence_images) == 1
                    )
                    return RespondAction(
                        content='The figure is useful.',
                        content_kind='coordination',
                        operation=Clarify(kind='clarify'),
                        cited_evidence_positions=[0],
                    )

            teacher = TeachingTeacherNode(
                Teacher(),
                TokenBudget(Counter(), token_limit=100000),
                0,
                image_budget=ImagePayloadBudget(
                    max_images=4, max_bytes=1000000, max_pixels=1000000
                ),
                discovery_preview_characters=320,
                research_enabled=False,
            )
            graph = TeachingInteractionGraph(
                teacher,
                teacher,
                teacher,
                TeachingInteractionToolsNode(None),
                max_tool_calls=0,
            )
            next_request = TeachingTurnRequest(
                user_uuid=user,
                session_uuid=session_uuid,
                expected_revision=2,
                snapshot=resumed,
                source_uuids=None,
            )
            next_result = await graph.run(
                TeachingInteractionRequest(
                    turn=next_request, learner_history=[]
                )
            )
            assert next_result.workspace.active_block_positions == [block_pos]
            assert next_result.workspace.active_image_positions == [image_pos]
            assert (
                next_result.workspace.image_registry[image_pos].data
                == image_bytes
            )
            await node.run(
                TeachingTurnState(request=next_request, interaction=next_result)
            )
            published = await snapshots.load_owned_snapshot(
                user, staged.source.uuid
            )
            assert published is not None and published.pages[0].index == 17
            reloaded = await teaching.load_session(user, session_uuid)
            assert reloaded.turns[-1].citations[0].uuid == block.uuid
        finally:
            async with database.session() as session:
                result = await session.run(
                    'MATCH (u:User {uuid:$user}) OPTIONAL MATCH (u)-[:HAS_TEACHING_SESSION]->(s) OPTIONAL MATCH (s)-[:HAS_TURN]->(t) DETACH DELETE t,s,u',
                    user=user,
                )
                await result.consume()
                result = await session.run(
                    'MATCH (s:Source {uuid:$uuid}) OPTIONAL MATCH (s)-[:HAS_PAGE]->(p) OPTIONAL MATCH (p)-[:CONTAINS_BLOCK]->(b) OPTIONAL MATCH (b)-[:CONTAINS_VISUAL_ASSET]->(a) DETACH DELETE a,b,p,s',
                    uuid=staged.source.uuid,
                )
                await result.consume()
            await database.close()

    asyncio.run(exercise())
