"""Real graph flow from source candidates through proposition reasoning."""

import asyncio
import os
from types import SimpleNamespace
from uuid import uuid4

import pytest

from kms.config.services import DatabaseSettings
from kms.core.tokenization.budget import TokenBudget
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.database.client import DatabaseClient
from kms.database.schema import ensure_structural_schema
from kms.knowledge.candidates import (
    ExistingPropositionReference,
    NewArgument,
    NewProposition,
    NewPropositionReference,
)
from kms.knowledge.commit import KnowledgeCommitNode, KnowledgeCommitter
from kms.knowledge.extraction.models import (
    UnitExtractionCandidates,
    UnitExtractionResult,
)
from kms.knowledge.models import Proposition
from kms.knowledge.reasoning.graph import KnowledgeReasoningGraph
from kms.knowledge.reasoning.models import ReasoningCandidates
from kms.knowledge.reasoning.node import KnowledgeReasoningNode
from kms.knowledge.retrieval.node import KnowledgeSearchNode
from kms.knowledge.retrieval.search import KnowledgeSearch
from kms.knowledge.retrieval.settings import KnowledgeSearchSettings
from kms.knowledge.storage import KnowledgeRepository
from kms.media.models import Material

_REQUIRED = (
    'KMS_DATABASE__URI',
    'KMS_DATABASE__USERNAME',
    'KMS_DATABASE__PASSWORD',
    'KMS_DATABASE__DATABASE',
)


class _Embedder:
    async def embed(self, request):
        return [[1.0, 0.0] for _ in request.materials]


class _Auditor:
    async def audit(self, user, candidates):
        by_id = {
            item.id: Proposition(name=item.name, statement=item.statement)
            for item in candidates
        }
        return SimpleNamespace(
            canonical=by_id, new_propositions=list(by_id.values())
        )


class _LinkAuditor:
    def __init__(self, existing):
        self.existing = existing

    async def audit(self, user, candidates):
        fresh = Proposition(
            name=candidates[2].name, statement=candidates[2].statement
        )
        return SimpleNamespace(
            canonical={
                candidates[0].id: self.existing,
                candidates[1].id: self.existing,
                candidates[2].id: fresh,
            },
            new_propositions=[fresh],
        )


class _Selection:
    async def select(self, *, query, material_kind, candidates):
        return list(range(len(candidates)))


class _Counter:
    def count_texts(self, texts):
        return [len(text) for text in texts]


class _Reasoner:
    async def acall(self, *, request):
        if not request.context.matched:
            return ReasoningCandidates(new_propositions=[], new_arguments=[])
        return ReasoningCandidates(
            new_propositions=[
                NewProposition(
                    id='derived', name='Derived', statement='A derived claim.'
                )
            ],
            new_arguments=[
                NewArgument(
                    premises=[
                        ExistingPropositionReference(
                            uuid=request.context.matched[0].uuid
                        )
                    ],
                    conclusion=NewPropositionReference(id='derived'),
                )
            ],
        )


@pytest.mark.skipif(
    os.getenv('KMS_NEO4J_IT') != '1'
    or not all(os.getenv(name) for name in _REQUIRED),
    reason='KMS Neo4j integration environment is not enabled',
)
def test_source_extraction_to_search_and_constructed_reasoning():
    async def exercise() -> None:
        database = DatabaseClient(
            DatabaseSettings(
                uri=os.environ[_REQUIRED[0]],
                username=os.environ[_REQUIRED[1]],
                password=os.environ[_REQUIRED[2]],
                database=os.environ[_REQUIRED[3]],
            )
        )
        prefix = f'knowledge-flow-{uuid4().hex}'
        owner, other = f'{prefix}-owner', f'{prefix}-other'
        source, unit = f'{prefix}-source', f'{prefix}-unit'
        try:
            await ensure_structural_schema(database.session)
            async with database.session() as session:
                result = await session.run(
                    'CREATE (u:User {uuid: $owner})-[:OWNS_SOURCE]->'
                    '(s:Source {uuid: $source})-[:HAS_UNIT]->'
                    '(:SourceUnit {uuid: $unit}), (:User {uuid: $other})',
                    owner=owner,
                    other=other,
                    source=source,
                    unit=unit,
                )
                await result.consume()
            repository = KnowledgeRepository(database.session)
            embedder = _Embedder()
            extracted = UnitExtractionResult(
                source_uuid=source,
                target_unit_uuid=unit,
                candidates=UnitExtractionCandidates(
                    new_propositions=[
                        NewProposition(
                            id='source-claim',
                            name='Claim',
                            statement='A source claim.',
                        )
                    ],
                    existing_proposition_uuids=[],
                    new_arguments=[],
                    existing_argument_uuids=[],
                ),
            )
            stored = await KnowledgeCommitter(
                repository,
                embedder,
                'test-model',
                auditor=_Auditor(),
            ).commit(owner, extracted)
            [premise] = stored.propositions

            search = KnowledgeSearch(
                repository,
                embedder,
                'test-model',
                selection=_Selection(),
                settings=KnowledgeSearchSettings(),
            )
            search_node = KnowledgeSearchNode(search)
            reasoning_node = KnowledgeReasoningNode(
                _Reasoner(),
                TokenBudget(MaterialTokenEstimator(_Counter()), 100000),
                candidate_limit=64,
                argument_limit=64,
            )
            committer = KnowledgeCommitter(
                repository,
                embedder,
                'test-model',
                auditor=_Auditor(),
            )
            graph = KnowledgeReasoningGraph(
                search_node,
                reasoning_node,
                KnowledgeCommitNode(committer),
            )
            result = await graph.reason(owner, 'Explain the claim')
            assert [item.uuid for item in result.context.matched] == [
                premise.uuid
            ]
            [conclusion] = result.stored.propositions
            assert conclusion.uuid not in {
                item.uuid for item in result.context.propositions
            }
            [argument] = result.stored.arguments
            assert argument.premise_uuids == [premise.uuid]
            assert argument.conclusion_uuid == conclusion.uuid

            context = await search.search(owner, Material.from_text('Explain'))
            assert {item.uuid for item in context.propositions} == {
                premise.uuid,
                conclusion.uuid,
            }
            assert [item.uuid for item in context.arguments] == [argument.uuid]
            foreign = await search.search(other, Material.from_text('Explain'))
            assert foreign.propositions == []
            async with database.session() as session:
                result = await session.run(
                    'MATCH (p:Proposition {uuid: $premise})-[:FROM_SOURCE]->(s:Source {uuid: $source}) '
                    'MATCH (a:Argument {uuid: $argument}) '
                    'RETURN a.origin AS origin',
                    premise=premise.uuid,
                    source=source,
                    argument=argument.uuid,
                )
                assert (await result.single(strict=True))[
                    'origin'
                ] == 'constructed'
            duplicate = UnitExtractionResult(
                source_uuid=source,
                target_unit_uuid=unit,
                candidates=UnitExtractionCandidates(
                    new_propositions=[
                        NewProposition(
                            id='a', name='Same', statement='Same claim'
                        ),
                        NewProposition(
                            id='b',
                            name='Same again',
                            statement='Same claim again',
                        ),
                        NewProposition(
                            id='c',
                            name='New conclusion',
                            statement='Another consequence',
                        ),
                    ],
                    existing_proposition_uuids=[],
                    new_arguments=[
                        NewArgument(
                            premises=[NewPropositionReference(id='b')],
                            conclusion=NewPropositionReference(id='c'),
                        )
                    ],
                    existing_argument_uuids=[],
                ),
            )
            merged = await KnowledgeCommitter(
                repository,
                embedder,
                'test-model',
                auditor=_LinkAuditor(premise),
            ).commit(owner, duplicate)
            assert [p.uuid for p in merged.propositions].count(
                premise.uuid
            ) == 1
            new_conclusion = next(
                p for p in merged.propositions if p.uuid != premise.uuid
            )
            assert merged.arguments[0].premise_uuids == [premise.uuid]
            assert merged.arguments[0].conclusion_uuid == new_conclusion.uuid
            async with database.session() as session:
                result = await session.run(
                    'MATCH (:User {uuid: $owner})-[:OWNS_PROPOSITION]->(p:Proposition) '
                    'OPTIONAL MATCH (p)-[r:FROM_SOURCE]->(:Source {uuid: $source}) '
                    'RETURN count(DISTINCT p) AS propositions, count(r) AS provenance',
                    owner=owner,
                    source=source,
                )
                row = await result.single(strict=True)
                assert row['propositions'] == 3
                assert row['provenance'] == 2
        finally:
            async with database.session() as session:
                result = await session.run(
                    'MATCH (u:User {uuid: $owner}) '
                    'OPTIONAL MATCH (u)-[:OWNS_PROPOSITION|OWNS_ARGUMENT]->(k) '
                    'DETACH DELETE k, u',
                    owner=owner,
                )
                await result.consume()
                result = await session.run(
                    'MATCH (n) WHERE n.uuid IN $ids DETACH DELETE n',
                    ids=[other, source, unit],
                )
                await result.consume()
            await database.close()

    asyncio.run(exercise())
