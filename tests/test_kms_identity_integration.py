"""Identity auditing and persistence with concurrent Neo4j transactions."""

import asyncio
import os
from uuid import uuid4

import pytest

from kms.config.services import DatabaseSettings
from kms.core.tokenization.budget import TokenBudget
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.database.client import DatabaseClient
from kms.database.schema import ensure_structural_schema
from kms.knowledge.candidates import NewProposition
from kms.knowledge.commit import KnowledgeCommitter
from kms.knowledge.extraction.models import (
    UnitExtractionCandidates,
    UnitExtractionResult,
)
from kms.knowledge.identity.audit import IdentityAuditor
from kms.knowledge.identity.models import IdentityDecision
from kms.knowledge.storage import KnowledgeRepository

_REQUIRED = (
    'KMS_DATABASE__URI',
    'KMS_DATABASE__USERNAME',
    'KMS_DATABASE__PASSWORD',
    'KMS_DATABASE__DATABASE',
)


class _Counter:
    def count_texts(self, texts):
        return [len(text) for text in texts]


class _Embedder:
    def __init__(self, *, fail_documents: bool = False) -> None:
        self.fail_documents = fail_documents
        self.input_types = []

    async def embed(self, request):
        self.input_types.append(request.input_type)
        if self.fail_documents and request.input_type == 'document':
            raise RuntimeError('embedding unavailable')
        return [[1.0, 0.0] for _ in request.materials]


def _identity_auditor(repository, embedder) -> IdentityAuditor:
    return IdentityAuditor(
        repository,
        embedder,
        'identity-test-model',
        _Identity(),
        TokenBudget(MaterialTokenEstimator(_Counter()), 100000),
        candidate_limit=8,
    )


def _committer(repository, embedder) -> KnowledgeCommitter:
    return KnowledgeCommitter(
        repository,
        embedder,
        'identity-test-model',
        auditor=_identity_auditor(repository, embedder),
    )


class _Identity:
    async def acall(self, *, request):
        return IdentityDecision(match_uuid=request.matches[0].uuid)


@pytest.mark.skipif(
    os.getenv('KMS_NEO4J_IT') != '1'
    or not all(os.getenv(name) for name in _REQUIRED),
    reason='KMS Neo4j integration environment is not enabled',
)
def test_concurrent_identical_candidates_commit_one_canonical_proposition():
    async def exercise():
        database = DatabaseClient(
            DatabaseSettings(
                uri=os.environ[_REQUIRED[0]],
                username=os.environ[_REQUIRED[1]],
                password=os.environ[_REQUIRED[2]],
                database=os.environ[_REQUIRED[3]],
            )
        )
        suffix = uuid4().hex
        owner, source, unit = (
            f'identity-{part}-{suffix}' for part in ('owner', 'source', 'unit')
        )
        try:
            await ensure_structural_schema(database.session)
            async with database.session() as session:
                result = await session.run(
                    'CREATE (:User {uuid:$owner})-[:OWNS_SOURCE]->(s:Source {uuid:$source})-[:HAS_UNIT]->(:SourceUnit {uuid:$unit})',
                    owner=owner,
                    source=source,
                    unit=unit,
                )
                await result.consume()
            repository = KnowledgeRepository(database.session)
            embedder = _Embedder()
            committer = _committer(repository, embedder)

            def extraction(local_id):
                return UnitExtractionResult(
                    source_uuid=source,
                    target_unit_uuid=unit,
                    candidates=UnitExtractionCandidates(
                        new_propositions=[
                            NewProposition(
                                id=local_id,
                                name='Claim',
                                statement='The same claim',
                            )
                        ],
                        existing_proposition_uuids=[],
                        new_arguments=[],
                        existing_argument_uuids=[],
                    ),
                )

            first, second = await asyncio.gather(
                committer.commit(owner, extraction('a')),
                committer.commit(owner, extraction('b')),
            )
            assert first.propositions[0].uuid == second.propositions[0].uuid
            async with database.session() as session:
                result = await session.run(
                    'MATCH (:User {uuid:$owner})-[:OWNS_PROPOSITION]->(p:Proposition) OPTIONAL MATCH (p)-[r:FROM_SOURCE]->(:Source {uuid:$source}) RETURN count(DISTINCT p) AS propositions, count(r) AS sources',
                    owner=owner,
                    source=source,
                )
                record = await result.single(strict=True)
                assert (record['propositions'], record['sources']) == (1, 1)
        finally:
            async with database.session() as session:
                result = await session.run(
                    'MATCH (u:User {uuid:$owner}) OPTIONAL MATCH (u)-[:OWNS_PROPOSITION|OWNS_ARGUMENT]->(p) DETACH DELETE p, u',
                    owner=owner,
                )
                await result.consume()
                result = await session.run(
                    'MATCH (n) WHERE n.uuid IN $ids DETACH DELETE n',
                    ids=[source, unit],
                )
                await result.consume()
            await database.close()

    asyncio.run(exercise())


@pytest.mark.skipif(
    os.getenv('KMS_NEO4J_IT') != '1'
    or not all(os.getenv(name) for name in _REQUIRED),
    reason='KMS Neo4j integration environment is not enabled',
)
def test_document_embedding_failure_rolls_back_and_releases_owner_lock():
    async def exercise():
        database = DatabaseClient(
            DatabaseSettings(
                uri=os.environ[_REQUIRED[0]],
                username=os.environ[_REQUIRED[1]],
                password=os.environ[_REQUIRED[2]],
                database=os.environ[_REQUIRED[3]],
            )
        )
        suffix = uuid4().hex
        owner, source, unit = (
            f'embedding-failure-{part}-{suffix}'
            for part in ('owner', 'source', 'unit')
        )
        candidate_name = f'Rollback candidate {suffix}'
        result = UnitExtractionResult(
            source_uuid=source,
            target_unit_uuid=unit,
            candidates=UnitExtractionCandidates(
                new_propositions=[
                    NewProposition(
                        id='candidate',
                        name=candidate_name,
                        statement='Must remain absent after failure',
                    )
                ],
                existing_proposition_uuids=[],
                new_arguments=[],
                existing_argument_uuids=[],
            ),
        )
        try:
            await ensure_structural_schema(database.session)
            async with database.session() as session:
                created = await session.run(
                    'CREATE (:User {uuid:$owner})-[:OWNS_SOURCE]->'
                    '(s:Source {uuid:$source})-[:HAS_UNIT]->'
                    '(:SourceUnit {uuid:$unit})',
                    owner=owner,
                    source=source,
                    unit=unit,
                )
                await created.consume()
            repository = KnowledgeRepository(database.session)
            embedder = _Embedder(fail_documents=True)
            committer = _committer(repository, embedder)

            with pytest.raises(RuntimeError, match='embedding unavailable'):
                await committer.commit(owner, result)
            assert embedder.input_types == ['query', 'document']

            async with database.session() as session:
                query = await session.run(
                    'MATCH (p:Proposition {name:$name}) '
                    'OPTIONAL MATCH (p)-[r:FROM_SOURCE]->'
                    '(:Source {uuid:$source}) '
                    'RETURN count(DISTINCT p) AS propositions, count(r) AS sources',
                    name=candidate_name,
                    source=source,
                )
                row = await query.single(strict=True)
                assert (row['propositions'], row['sources']) == (0, 0)

            embedder.fail_documents = False
            async with asyncio.timeout(5):
                stored = await committer.commit(owner, result)
            assert [item.name for item in stored.propositions] == [
                candidate_name
            ]
            assert embedder.input_types == [
                'query',
                'document',
                'query',
                'document',
            ]
        finally:
            async with database.session() as session:
                deleted = await session.run(
                    'MATCH (u:User {uuid:$owner}) '
                    'OPTIONAL MATCH (u)-[:OWNS_PROPOSITION|OWNS_ARGUMENT]->(p) '
                    'DETACH DELETE p, u',
                    owner=owner,
                )
                await deleted.consume()
                deleted = await session.run(
                    'MATCH (n) WHERE n.uuid IN $ids DETACH DELETE n',
                    ids=[source, unit],
                )
                await deleted.consume()
            await database.close()

    asyncio.run(exercise())
