"""Atomic publication of only teaching-used staged sources."""

import asyncio
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from neo4j.exceptions import ResultNotSingleError

from kms2.config.services import DatabaseSettings
from kms2.config.settings import Settings
from kms2.core.model.block import SourceBlock
from kms2.core.model.block_types import BlockType
from kms2.core.model.knowledge.admission import (
    KnowledgeAdmission,
    KnowledgeAssertionWrite,
)
from kms2.core.model.knowledge.proposition import (
    KnowledgeAssertion,
    Proposition,
)
from kms2.core.model.page import SourcePage
from kms2.core.model.source import Source, SourceRevisionReference
from kms2.core.model.teaching.operation import Clarify, RespondAction
from kms2.core.model.teaching.workflow import (
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
from kms2.database import schema
from kms2.database.client import DatabaseClient
from kms2.database.source.source_snapshot_repository import (
    SourceSnapshotRepository,
)
from kms2.database.teaching.repository import TeachingRepository
from kms2.langgraph.teaching.state import TeachingTurnState
from kms2.node.teaching.persistence import TeachingPersistenceNode
from kms2.service.teaching import TeachingService

_REQUIRED_ENVIRONMENT = (
    'KMS2_DATABASE__URI',
    'KMS2_DATABASE__USERNAME',
    'KMS2_DATABASE__PASSWORD',
    'KMS2_DATABASE__DATABASE',
)


def _staged(key: str, *, asset: VisualAsset | None = None):
    source = Source(
        uuid=str(uuid4()),
        key=key,
        metadata={
            'capture_scope': 'selected_pdf_pages',
            'selected_page_indexes': [17],
        },
    )
    block = SourceBlock(
        block_type=BlockType.IMAGE if asset else BlockType.PARAGRAPH,
        content=key,
        assets=[asset] if asset else [],
    )
    return finalize_snapshot_identities(
        source, [SourcePage(index=17, markdown=key, blocks=[block])]
    )


def _state(snapshot, user_uuid, workspace, citations, knowledge, revision):
    interaction = TeachingInteractionResult(
        response=RespondAction(
            content='A source-backed explanation.',
            content_kind='instruction',
            operation=Clarify(kind='clarify'),
            cited_evidence_positions=citations,
        ),
        workspace=workspace,
        visible_turn_indexes=[],
        visible_observation_indexes=[],
        visible_citation_receipt_indexes=list(range(len(workspace.receipts))),
    )
    return TeachingTurnState(
        request=TeachingTurnRequest(
            user_uuid=user_uuid,
            session_uuid=snapshot.session.uuid,
            expected_revision=revision,
            snapshot=snapshot,
            source_uuids=None,
        ),
        interaction=interaction,
        knowledge_admission=knowledge,
    )


def _knowledge(source, *, revision: int):
    now = datetime(2026, 1, 1, tzinfo=UTC)
    proposition = Proposition(
        uuid=str(uuid4()),
        content='Supported claim',
        embedding=[1.0, 0.0],
        created_at=now,
    )
    assertion = KnowledgeAssertion(uuid=str(uuid4()), created_at=now)
    admission = KnowledgeAdmission(
        expected_revision=revision,
        source_revisions=[
            SourceRevisionReference(
                source_uuid=source.source.uuid, revision=source.source.revision
            )
        ],
        new_propositions=[proposition],
        proposition_uuids=[proposition.uuid],
        assertions=[
            KnowledgeAssertionWrite(
                assertion=assertion,
                proposition_uuid=proposition.uuid,
                supporting_block_uuids=[source.pages[0].blocks[0].uuid],
                reasoning_step_uuids=[],
            )
        ],
        new_reasoning_steps=[],
        existing_reasoning_step_uuids=[],
    )
    return admission, proposition


@pytest.mark.skipif(
    os.getenv('KMS2_NEO4J_IT') != '1'
    or not all(os.getenv(name) for name in _REQUIRED_ENVIRONMENT),
    reason='KMS2 Neo4j integration environment is not enabled',
)
def test_used_staged_sources_publish_atomically_and_unused_sources_do_not(
    tmp_path: Path,
):
    async def exercise():
        settings = Settings(
            _env_file=None,
            database=DatabaseSettings(
                uri=os.environ[_REQUIRED_ENVIRONMENT[0]],
                username=os.environ[_REQUIRED_ENVIRONMENT[1]],
                password=os.environ[_REQUIRED_ENVIRONMENT[2]],
                database=os.environ[_REQUIRED_ENVIRONMENT[3]],
            ),
        )
        database = DatabaseClient(settings.database)
        repository = SourceSnapshotRepository(database.session)
        teaching = TeachingService(TeachingRepository(database.session))
        node = TeachingPersistenceNode(teaching)
        user_uuid = str(uuid4())
        image_path = tmp_path / 'figure.png'
        image_path.write_bytes(b'original image')
        cited = _staged('cited block')
        image = _staged('cited image', asset=VisualAsset(path=str(image_path)))
        support = _staged('support only')
        unused = _staged('inspected but unused')
        owned = _staged('already owned')
        stale_source = _staged('stale source')
        sources = [cited, image, support, unused, owned, stale_source]
        try:
            await schema.ensure_structural_schema(database.session)
            async with database.session() as session:
                result = await session.run(
                    'CREATE (:User {uuid: $uuid})', uuid=user_uuid
                )
                await result.consume()
            owned_snapshot = await repository.create_owned_snapshot(
                user_uuid, owned.source, owned.pages
            )
            created = await teaching.start_session(
                user_uuid, 'Learn from evidence'
            )
            session_uuid = created.session.uuid
            snapshot = await teaching.load_snapshot(user_uuid, session_uuid)
            image_block = image.pages[0].blocks[0]
            workspace = TeachingEvidenceWorkspace(
                staged_sources=[cited, image, support, unused],
                acquired_sources=[
                    cited,
                    image,
                    support,
                    unused,
                    owned_snapshot,
                ],
                receipts=[
                    BlockInspection(
                        source=cited.source,
                        block=cited.pages[0].blocks[0],
                        page_indexes=[17],
                    ),
                    ImageInspection(
                        source=image.source,
                        asset=image_block.assets[0],
                        block_uuids=[image_block.uuid],
                        data=b'original image',
                    ),
                    BlockInspection(
                        source=unused.source,
                        block=unused.pages[0].blocks[0],
                        page_indexes=[17],
                    ),
                ],
            )
            knowledge, proposition = _knowledge(support, revision=0)
            result = await node.run(
                _state(snapshot, user_uuid, workspace, [0, 1], knowledge, 0)
            )
            assert result['final_view'].session.revision == 1
            for item in (cited, image, support, owned):
                loaded = await repository.load_owned_snapshot(
                    user_uuid, item.source.uuid
                )
                assert loaded is not None
                assert loaded.pages[0].index == 17
                assert (
                    loaded.source.metadata['capture_scope']
                    == 'selected_pdf_pages'
                )
            assert (
                await repository.load_owned_snapshot(
                    user_uuid, unused.source.uuid
                )
                is None
            )
            citations = result['final_view'].turns[-1].citations
            assert {(item.kind, item.uuid) for item in citations} == {
                ('block', cited.pages[0].blocks[0].uuid),
                ('image', image_block.assets[0].uuid),
            }
            async with database.session() as session:
                records = await session.run(
                    """
                    MATCH (:User {uuid: $user})-[:HAS_TEACHING_SESSION]->
                          (:TeachingSession {uuid: $session})-[:HAS_TURN]->
                          (:TeachingTurn)-[:HAS_KNOWLEDGE_ASSERTION]->
                          (assertion:KnowledgeAssertion)-[:ASSERTS]->
                          (:Proposition {uuid: $proposition})
                    MATCH (assertion)-[:SUPPORTED_BY]->(block:SourceBlock)
                    RETURN collect(block.uuid) AS blocks
                    """,
                    user=user_uuid,
                    session=session_uuid,
                    proposition=proposition.uuid,
                )
                row = await records.single(strict=True)
                assert row['blocks'] == [support.pages[0].blocks[0].uuid]

            no_use = await teaching.start_session(user_uuid, 'No source used')
            no_use_snapshot = await teaching.load_snapshot(
                user_uuid, no_use.session.uuid
            )
            no_use_workspace = TeachingEvidenceWorkspace(
                staged_sources=[unused],
                acquired_sources=[unused, owned_snapshot],
            )
            no_use_result = await node.run(
                _state(
                    no_use_snapshot, user_uuid, no_use_workspace, [], None, 0
                )
            )
            assert no_use_result['final_view'].session.revision == 1
            assert (
                await repository.load_owned_snapshot(
                    user_uuid, unused.source.uuid
                )
                is None
            )

            async with database.session() as session:
                result = await session.run(
                    'MATCH (s:Source {uuid: $uuid}) SET s.revision = 1',
                    uuid=owned.source.uuid,
                )
                await result.consume()
            changed_owned = await teaching.start_session(
                user_uuid, 'Changed owned source'
            )
            changed_snapshot = await teaching.load_snapshot(
                user_uuid, changed_owned.session.uuid
            )
            with pytest.raises(ResultNotSingleError):
                await node.run(
                    _state(
                        changed_snapshot,
                        user_uuid,
                        no_use_workspace,
                        [],
                        None,
                        0,
                    )
                )
            assert (
                await teaching.load_snapshot(
                    user_uuid, changed_owned.session.uuid
                )
            ).turns == []

            other = await teaching.start_session(user_uuid, 'Stale turn')
            stale_snapshot = await teaching.load_snapshot(
                user_uuid, other.session.uuid
            )
            stale_workspace = TeachingEvidenceWorkspace(
                staged_sources=[stale_source],
                acquired_sources=[stale_source],
                receipts=[
                    BlockInspection(
                        source=stale_source.source,
                        block=stale_source.pages[0].blocks[0],
                        page_indexes=[17],
                    )
                ],
            )
            stale_knowledge, stale_proposition = _knowledge(
                stale_source, revision=1
            )
            with pytest.raises(ResultNotSingleError):
                await node.run(
                    _state(
                        stale_snapshot,
                        user_uuid,
                        stale_workspace,
                        [0],
                        stale_knowledge,
                        99,
                    )
                )
            assert (
                await repository.load_owned_snapshot(
                    user_uuid, stale_source.source.uuid
                )
                is None
            )
            reloaded = await teaching.load_session(
                user_uuid, other.session.uuid
            )
            assert reloaded.turns == []
            async with database.session() as session:
                result = await session.run(
                    'MATCH (p:Proposition {uuid: $uuid}) RETURN count(p) AS total',
                    uuid=stale_proposition.uuid,
                )
                assert (await result.single(strict=True))['total'] == 0
        finally:
            async with database.session() as session:
                result = await session.run(
                    """
                    MATCH (user:User {uuid: $user})
                    OPTIONAL MATCH (user)-[:HAS_TEACHING_SESSION]->(session:TeachingSession)
                    OPTIONAL MATCH (session)-[:HAS_TURN]->(turn:TeachingTurn)
                    OPTIONAL MATCH (turn)-[:HAS_KNOWLEDGE_ASSERTION]->(assertion:KnowledgeAssertion)
                    OPTIONAL MATCH (user)-[:HAS_PROPOSITION]->(proposition:Proposition)
                    DETACH DELETE assertion, turn, session, proposition, user
                    """,
                    user=user_uuid,
                )
                await result.consume()
                result = await session.run(
                    """
                    UNWIND $sources AS uuid
                    MATCH (source:Source {uuid: uuid})
                    OPTIONAL MATCH (source)-[:HAS_PAGE]->(page:SourcePage)
                    OPTIONAL MATCH (page)-[:CONTAINS_BLOCK]->(block:SourceBlock)
                    OPTIONAL MATCH (block)-[:CONTAINS_VISUAL_ASSET]->(asset:VisualAsset)
                    DETACH DELETE asset, block, page, source
                    """,
                    sources=[item.source.uuid for item in sources],
                )
                await result.consume()
            await database.close()

    asyncio.run(exercise())
