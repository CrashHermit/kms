"""Real-driver checks for atomic teaching and knowledge commits."""

import asyncio
import os
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from neo4j.exceptions import ResultNotSingleError

from kms import application
from kms.config.services import DatabaseSettings
from kms.config.settings import Settings
from kms.core.model.block import SourceBlock
from kms.core.model.block_types import BlockType
from kms.core.model.knowledge.admission import (
    KnowledgeAdmission,
    KnowledgeAssertionWrite,
)
from kms.core.model.knowledge.argument import ArgumentRecord
from kms.core.model.knowledge.proposition import (
    Argument,
    KnowledgeAssertion,
    Proposition,
)
from kms.core.model.page import SourcePage
from kms.core.model.source import Source, SourceRevisionReference
from kms.core.model.teaching.session import (
    TeachingPhase,
    TeachingTurnCommit,
)
from kms.database import schema
from kms.database.client import DatabaseClient
from kms.database.knowledge.repository import KnowledgeRepository
from kms.database.source.source_graph_repository import (
    SourceGraphRepository,
)
from kms.database.teaching.repository import TeachingRepository
from kms.inference.embedding.budget import VOYAGE_EMBEDDING_DIMENSION
from kms.service.teaching import TeachingService

_REQUIRED_ENVIRONMENT = (
    'KMS_DATABASE__URI',
    'KMS_DATABASE__USERNAME',
    'KMS_DATABASE__PASSWORD',
    'KMS_DATABASE__DATABASE',
)

pytestmark = pytest.mark.skipif(
    os.getenv('KMS_NEO4J_IT') != '1'
    or not all(os.getenv(name) for name in _REQUIRED_ENVIRONMENT),
    reason='KMS Neo4j integration environment is not enabled',
)


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        database=DatabaseSettings(
            uri=os.environ['KMS_DATABASE__URI'],
            username=os.environ['KMS_DATABASE__USERNAME'],
            password=os.environ['KMS_DATABASE__PASSWORD'],
            database=os.environ['KMS_DATABASE__DATABASE'],
        ),
    )


def test_teacher_turn_and_grounded_knowledge_commit_atomically() -> None:
    async def exercise() -> None:
        settings = _settings()
        database = DatabaseClient(settings.database)
        suffix = uuid4().hex
        user = None
        user_uuid = f'{suffix}-user'
        session_uuid = f'{suffix}-session'
        source_uuid = f'{suffix}-source'
        second_source_uuid = f'{suffix}-second-source'
        second_block_uuid = f'{suffix}-second-block'
        second_assertion_uuid = f'{suffix}-second-assertion'
        block_uuid = f'{suffix}-block'
        premise_uuid = f'{suffix}-premise'
        second_premise_uuid = f'{suffix}-second-premise'
        conclusion_uuid = f'{suffix}-conclusion'
        argument_uuid = f'{suffix}-argument'
        assertion_uuids = [f'{suffix}-assertion-{index}' for index in range(3)]
        session_uuids: list[str] = []
        all_uuids = [
            session_uuid,
            source_uuid,
            block_uuid,
            premise_uuid,
            second_premise_uuid,
            conclusion_uuid,
            argument_uuid,
            *assertion_uuids,
            second_source_uuid,
            second_block_uuid,
            second_assertion_uuid,
        ]
        try:
            await schema.ensure_structural_schema(database.session)
            user = await application.create_user(
                settings,
                f'Knowledge integration {suffix}',
            )
            user_uuid = user.uuid
            all_uuids.append(user_uuid)
            async with database.session() as session:
                result = await session.run(
                    """
                    MATCH (user:User {uuid: $user_uuid})
                    CREATE (source:Source {
                        uuid: $source_uuid,
                        key: 'grounded-fixture',
                        metadata: '{}',
                        revision: 0
                    })
                    CREATE (user)-[:OWNS_SOURCE]->(source)
                    CREATE (page:SourcePage {
                        index: 4,
                        markdown: 'Captured evidence'
                    })
                    CREATE (source)-[:HAS_PAGE]->(page)
                    CREATE (block:SourceBlock {
                        uuid: $block_uuid,
                        block_type: 'text',
                        content: 'Every even integer is divisible by two.'
                    })
                    CREATE (page)-[:CONTAINS_BLOCK]->(block)
                    CREATE (second_source:Source {
                        uuid: $second_source_uuid,
                        key: 'grounded-second-fixture',
                        metadata: '{}',
                        revision: 0
                    })
                    CREATE (user)-[:OWNS_SOURCE]->(second_source)
                    CREATE (second_page:SourcePage {
                        index: 9,
                        markdown: 'Independent captured evidence'
                    })
                    CREATE (second_source)-[:HAS_PAGE]->(second_page)
                    CREATE (second_block:SourceBlock {
                        uuid: $second_block_uuid,
                        block_type: 'text',
                        content: 'Independent corroboration.'
                    })
                    CREATE (second_page)-[:CONTAINS_BLOCK]->(second_block)
                    RETURN source.uuid AS uuid
                    """,
                    user_uuid=user_uuid,
                    source_uuid=source_uuid,
                    block_uuid=block_uuid,
                    second_source_uuid=second_source_uuid,
                    second_block_uuid=second_block_uuid,
                )
                await result.single(strict=True)
            now = datetime.now(UTC)
            dimensions = VOYAGE_EMBEDDING_DIMENSION
            premise = Proposition(
                uuid=premise_uuid,
                content='Every even integer is divisible by two.',
                embedding=[0.01] * dimensions,
                created_at=now,
            )
            second_premise = Proposition(
                uuid=second_premise_uuid,
                content='The integer 8 is even.',
                embedding=[0.015] * dimensions,
                created_at=now,
            )
            conclusion = Proposition(
                uuid=conclusion_uuid,
                content='The integer 8 is divisible by two.',
                embedding=[0.02] * dimensions,
                created_at=now,
            )
            argument = Argument(
                uuid=argument_uuid,
                explanation='Eight is even, so the stated rule applies.',
                embedding=[0.025] * dimensions,
                created_at=now,
            )
            source_revision = SourceRevisionReference(
                source_uuid=source_uuid,
                revision=0,
            )
            admission = KnowledgeAdmission(
                expected_revision=0,
                source_revisions=[source_revision],
                new_propositions=[premise, second_premise, conclusion],
                proposition_uuids=[
                    premise_uuid,
                    second_premise_uuid,
                    conclusion_uuid,
                ],
                assertions=[
                    KnowledgeAssertionWrite(
                        assertion=KnowledgeAssertion(
                            uuid=assertion_uuids[0], created_at=now
                        ),
                        proposition_uuid=premise_uuid,
                        supporting_block_uuids=[block_uuid],
                        argument_uuids=[],
                    ),
                    KnowledgeAssertionWrite(
                        assertion=KnowledgeAssertion(
                            uuid=assertion_uuids[1], created_at=now
                        ),
                        proposition_uuid=second_premise_uuid,
                        supporting_block_uuids=[block_uuid],
                        argument_uuids=[],
                    ),
                    KnowledgeAssertionWrite(
                        assertion=KnowledgeAssertion(
                            uuid=assertion_uuids[2], created_at=now
                        ),
                        proposition_uuid=conclusion_uuid,
                        supporting_block_uuids=[block_uuid],
                        argument_uuids=[argument_uuid],
                    ),
                ],
                new_arguments=[
                    ArgumentRecord(
                        argument=argument,
                        premise_uuids=[premise_uuid, second_premise_uuid],
                        conclusion_uuid=conclusion_uuid,
                        supporting_block_uuids=[block_uuid],
                    )
                ],
                existing_argument_uuids=[],
            )
            service = TeachingService(TeachingRepository(database.session))
            started = await service.start_session(user_uuid, 'Explain parity')
            session_uuid = started.session.uuid
            all_uuids.append(session_uuid)
            session_uuids.append(session_uuid)
            result = await service.record_result(
                user_uuid,
                session_uuid,
                'Eight is divisible by two because it is even.',
                TeachingTurnCommit(
                    phase=TeachingPhase.TEACHING,
                    current_assessment_uuid=None,
                    current_plan_uuid=None,
                    current_step_uuid=None,
                    pending_question_uuid=None,
                    source_revisions=[source_revision],
                    knowledge=admission,
                ),
                expected_revision=0,
            )
            assert result.session.revision == 1
            assert len(result.turns[-1].knowledge_assertions) == 3
            assert {
                item.proposition_uuid
                for item in result.turns[-1].knowledge_assertions
            } == {premise_uuid, second_premise_uuid, conclusion_uuid}
            assert all(
                item.teaching_turn_uuid == result.turns[-1].turn.uuid
                for item in result.turns[-1].knowledge_assertions
            )
            persisted_arguments = await KnowledgeRepository(
                database.session
            ).read_arguments(user_uuid, [argument_uuid])
            assert len(persisted_arguments) == 1
            assert (
                persisted_arguments[0].argument.embedding == argument.embedding
            )
            assert persisted_arguments[0].premise_uuids == sorted(
                [premise_uuid, second_premise_uuid]
            )
            assert persisted_arguments[0].supporting_block_uuids == [block_uuid]
            async with database.session() as session:
                rows = await session.run(
                    """
                    MATCH (user:User {uuid: $user_uuid})
                          -[:HAS_PROPOSITION]->(proposition:Proposition)
                    OPTIONAL MATCH (proposition)<-[:ASSERTS]-
                        (assertion:KnowledgeAssertion)
                        <-[:HAS_KNOWLEDGE_ASSERTION]-(turn:TeachingTurn)
                    OPTIONAL MATCH (assertion)-[:SUPPORTED_BY]->(block:SourceBlock)
                    OPTIONAL MATCH (argument:Argument)-[:PROVES]->
                        (proposition)
                    OPTIONAL MATCH (premise:Proposition)-[:PREMISE_OF]->
                        (argument)
                    WHERE proposition.uuid IN $proposition_uuids
                    RETURN proposition.uuid AS proposition_uuid,
                           collect(DISTINCT assertion.uuid) AS assertions,
                           collect(DISTINCT turn.uuid) AS turns,
                           collect(DISTINCT block.uuid) AS support_blocks,
                           collect(DISTINCT argument.uuid) AS arguments,
                           collect(DISTINCT premise.uuid) AS premises
                    ORDER BY proposition_uuid
                    """,
                    user_uuid=user_uuid,
                    proposition_uuids=[
                        premise_uuid,
                        second_premise_uuid,
                        conclusion_uuid,
                    ],
                )
                persisted = await rows.data()
                revision_result = await session.run(
                    'MATCH (user:User {uuid: $uuid}) '
                    'RETURN user.knowledge_revision AS revision',
                    uuid=user_uuid,
                )
                revision = await revision_result.single(strict=True)
                property_result = await session.run(
                    'MATCH (node) WHERE node.uuid IN $uuids '
                    'RETURN node.uuid AS uuid, keys(node) AS keys',
                    uuids=[
                        premise_uuid,
                        second_premise_uuid,
                        conclusion_uuid,
                        argument_uuid,
                    ],
                )
                node_properties = await property_result.data()
            by_uuid = {row['proposition_uuid']: row for row in persisted}
            assert set(by_uuid) == {
                premise_uuid,
                second_premise_uuid,
                conclusion_uuid,
            }
            assert by_uuid[premise_uuid]['support_blocks'] == [block_uuid]
            assert by_uuid[second_premise_uuid]['support_blocks'] == [
                block_uuid
            ]
            assert by_uuid[conclusion_uuid]['support_blocks'] == [block_uuid]
            assert set(by_uuid[conclusion_uuid]['premises']) == {
                premise_uuid,
                second_premise_uuid,
            }
            assert revision['revision'] == 1
            assert {row['uuid'] for row in node_properties} == {
                premise_uuid,
                second_premise_uuid,
                conclusion_uuid,
                argument_uuid,
            }
            assert all(
                not {
                    'user_uuid',
                    'source_uuid',
                    'session_uuid',
                    'turn_uuid',
                    'teaching_turn_uuid',
                    'parent_uuid',
                }.intersection(row['keys'])
                for row in node_properties
            )
            source_repository = SourceGraphRepository(database.session)
            with pytest.raises(
                ValueError, match='Source evidence is referenced'
            ):
                await source_repository.replace_source(
                    Source(uuid=source_uuid, key='grounded-fixture'),
                    [
                        SourcePage(
                            index=4,
                            markdown='Changed evidence',
                            blocks=[
                                SourceBlock(
                                    uuid=f'{suffix}-replacement-block',
                                    block_type=BlockType.PARAGRAPH,
                                    content='Changed evidence',
                                )
                            ],
                        )
                    ],
                )
            with pytest.raises(
                ValueError, match='Source evidence is referenced'
            ):
                await source_repository.replace_pedagogical_material(
                    source_uuid,
                    [
                        SourcePage(
                            index=4,
                            markdown='Captured evidence',
                            blocks=[
                                SourceBlock(
                                    uuid=block_uuid,
                                    block_type=BlockType.PARAGRAPH,
                                    content='Every even integer is divisible by two.',
                                )
                            ],
                        )
                    ],
                    [],
                    [],
                    [],
                )
            async with database.session() as session:
                result = await session.run(
                    """
                    MATCH (source:Source {uuid: $source_uuid})
                    MATCH (source)-[:HAS_PAGE]->(:SourcePage)
                          -[:CONTAINS_BLOCK]->(block:SourceBlock {
                              uuid: $block_uuid
                          })
                    RETURN coalesce(source.revision, 0) AS revision,
                           block.content AS content
                    """,
                    source_uuid=source_uuid,
                    block_uuid=block_uuid,
                )
                unchanged = await result.single(strict=True)
            assert unchanged['revision'] == 0
            assert (
                unchanged['content']
                == 'Every even integer is divisible by two.'
            )

            second_source_revision = SourceRevisionReference(
                source_uuid=second_source_uuid,
                revision=0,
            )
            reuse = KnowledgeAdmission(
                expected_revision=1,
                source_revisions=[second_source_revision],
                new_propositions=[],
                proposition_uuids=[conclusion_uuid],
                assertions=[
                    KnowledgeAssertionWrite(
                        assertion=KnowledgeAssertion(
                            uuid=second_assertion_uuid,
                            created_at=datetime.now(UTC),
                        ),
                        proposition_uuid=conclusion_uuid,
                        supporting_block_uuids=[second_block_uuid],
                        argument_uuids=[],
                    )
                ],
                new_arguments=[],
                existing_argument_uuids=[],
            )
            second_session = await service.start_session(
                user_uuid,
                'Reuse a grounded proposition',
            )
            all_uuids.append(second_session.session.uuid)
            session_uuids.append(second_session.session.uuid)
            second_turns = await service.record_result(
                user_uuid,
                second_session.session.uuid,
                'The same claim is supported by independent evidence.',
                TeachingTurnCommit(
                    phase=TeachingPhase.TEACHING,
                    current_assessment_uuid=None,
                    current_plan_uuid=None,
                    current_step_uuid=None,
                    pending_question_uuid=None,
                    source_revisions=[second_source_revision],
                    knowledge=reuse,
                ),
                expected_revision=0,
            )
            assert len(second_turns.turns[-1].knowledge_assertions) == 1
            assert (
                second_turns.turns[-1].knowledge_assertions[0].proposition_uuid
                == conclusion_uuid
            )
            async with database.session() as session:
                result = await session.run(
                    """
                    MATCH (user:User {uuid: $user_uuid})
                          -[:HAS_PROPOSITION]->(
                              proposition:Proposition {uuid: $proposition_uuid}
                          )
                    MATCH (assertion:KnowledgeAssertion)-[:ASSERTS]->(proposition)
                    MATCH (turn:TeachingTurn)-[:HAS_KNOWLEDGE_ASSERTION]->
                          (assertion)
                    OPTIONAL MATCH (assertion)-[:SUPPORTED_BY]->(block:SourceBlock)
                    RETURN assertion.uuid AS assertion_uuid,
                           turn.uuid AS turn_uuid,
                           collect(DISTINCT block.uuid) AS support_blocks
                    ORDER BY assertion_uuid
                    """,
                    user_uuid=user_uuid,
                    proposition_uuid=conclusion_uuid,
                )
                conclusion_assertions = await result.data()
            support_by_assertion = {
                row['assertion_uuid']: row['support_blocks']
                for row in conclusion_assertions
            }
            assert support_by_assertion == {
                assertion_uuids[2]: [block_uuid],
                second_assertion_uuid: [second_block_uuid],
            }
            with pytest.raises(ResultNotSingleError):
                await service.record_result(
                    user_uuid,
                    session_uuid,
                    'This stale teacher write must not persist.',
                    TeachingTurnCommit(
                        phase=TeachingPhase.TEACHING,
                        current_assessment_uuid=None,
                        current_plan_uuid=None,
                        current_step_uuid=None,
                        pending_question_uuid=None,
                        source_revisions=[source_revision],
                        knowledge=admission,
                    ),
                    expected_revision=0,
                )
            reloaded = await service.load_snapshot(user_uuid, session_uuid)
            assert reloaded.session.revision == 1
            assert len(reloaded.turns) == 1
        finally:
            async with database.session() as session:
                result = await session.run(
                    """
                    MATCH (session:TeachingSession)
                    WHERE session.uuid IN $session_uuids
                    OPTIONAL MATCH (session)-[:HAS_TURN]->(turn:TeachingTurn)
                    OPTIONAL MATCH (turn)-[:ASKS]->(question:TeachingQuestion)
                    OPTIONAL MATCH (session)-[:HAS_PLAN]->(plan:TeachingPlan)
                    OPTIONAL MATCH (plan)-[:HAS_STEP]->(step:TeachingPlanStep)
                    OPTIONAL MATCH (session)-[:HAS_ASSESSMENT]->
                        (assessment:ProbeAssessment)
                    WITH collect(DISTINCT session) + collect(DISTINCT turn)
                         + collect(DISTINCT question) + collect(DISTINCT plan)
                         + collect(DISTINCT step) + collect(DISTINCT assessment)
                         AS nodes
                    UNWIND nodes AS node
                    WITH node WHERE node IS NOT NULL
                    DETACH DELETE node
                    """,
                    session_uuids=session_uuids,
                )
                await result.consume()
                result = await session.run(
                    """
                    MATCH (source:Source)
                    WHERE source.uuid IN $source_uuids
                    OPTIONAL MATCH (source)-[:HAS_PAGE]->(page:SourcePage)
                    OPTIONAL MATCH (page)-[:CONTAINS_BLOCK]->(block:SourceBlock)
                    WITH collect(DISTINCT source)
                         + collect(DISTINCT page)
                         + collect(DISTINCT block) AS nodes
                    UNWIND nodes AS node
                    WITH node WHERE node IS NOT NULL
                    DETACH DELETE node
                    """,
                    source_uuids=[source_uuid, second_source_uuid],
                )
                await result.consume()
                result = await session.run(
                    """
                    MATCH (user:User {uuid: $user_uuid})
                    OPTIONAL MATCH (user)-[:HAS_DECK]->(deck:Deck)
                    OPTIONAL MATCH (user)-[:HAS_SCHEDULER_SETTINGS]->
                        (scheduler:UserSchedulerSettings)
                    WITH collect(user) + collect(deck) + collect(scheduler)
                        AS nodes
                    UNWIND nodes AS node
                    WITH node WHERE node IS NOT NULL
                    DETACH DELETE node
                    """,
                    user_uuid=user_uuid,
                )
                await result.consume()
                result = await session.run(
                    'MATCH (node) WHERE node.uuid IN $uuids DETACH DELETE node',
                    uuids=all_uuids,
                )
                await result.consume()
            await database.close()

    asyncio.run(exercise())
