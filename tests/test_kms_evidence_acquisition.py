"""Behavioral tests for isolated evidence acquisition and inspection."""

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from kms.core.media_budget import ImagePayloadBudget
from kms.core.model.block import SourceBlock
from kms.core.model.block_types import BlockType
from kms.core.model.knowledge.argument import ArgumentRecord
from kms.core.model.knowledge.proposition import (
    Argument,
    Proposition,
    PropositionRecord,
)
from kms.core.model.knowledge.retrieval import ArgumentSearchHit
from kms.core.model.source import Source
from kms.core.model.source_evidence import (
    SourceSearchHit,
    SourceStatementSearchHit,
)
from kms.core.model.source_semantic.source_statement import SourceStatement
from kms.core.model.teaching.acquisition import (
    AcquireEvidenceAction,
    LocalEvidenceRequest,
)
from kms.core.model.teaching.inference import (
    ReadArgumentsAction,
    ReadLearnerObservationsAction,
    ReadPropositionsAction,
    SearchArgumentsAction,
    SearchLearnerObservationsAction,
    SearchStatementsAction,
    TeachingToolObservation,
)
from kms.core.model.teaching.observation import (
    LearnerObservation,
    LearnerObservationEvidence,
    LearnerObservationKind,
    LearnerObservationRecord,
    LearnerObservationSearchHit,
)
from kms.core.model.teaching.workflow import EvidenceExecutionRequest
from kms.core.model.teaching.workspace import TeachingEvidenceWorkspace
from kms.node.evidence_acquisition.acquisition import EvidenceAcquisitionNode
from kms.node.teaching.evidence import register_proposition
from kms.node.teaching.tools import TeachingEvidenceTools
from kms.service.source_workspace import SourceWorkspaceService


class _SourceSearch:
    def __init__(self, hits=(), result_sets=None):
        self.hits = list(hits)
        self.result_sets = (
            [list(result) for result in result_sets]
            if result_sets is not None
            else None
        )
        self.calls = []

    async def search_sources(self, user_uuid, query, *, source_uuids=None):
        self.calls.append(('source', query, source_uuids))
        call_index = sum(call[0] == 'source' for call in self.calls) - 1
        hits = (
            self.result_sets[call_index]
            if self.result_sets is not None
            else self.hits
        )
        if source_uuids is None:
            return hits
        return [hit for hit in hits if hit.source.uuid in source_uuids]

    async def search_statements(self, *args, **kwargs):
        return []

    async def search_procedures(self, *args, **kwargs):
        return []


class _KnowledgeSearch:
    def __init__(self, propositions=(), arguments=()):
        self.propositions = list(propositions)
        self.arguments = list(arguments)
        self.calls = []

    async def search_propositions(self, user_uuid, query):
        self.calls.append(('proposition', query))
        return [SimpleNamespace(proposition=item) for item in self.propositions]

    async def search_arguments(self, user_uuid, query):
        self.calls.append(('argument', query))
        return self.arguments


class _SourceEvidence:
    async def read_passages(self, *args, **kwargs):
        raise AssertionError('Unexpected source passage read during search')

    async def read_images_at_revisions(self, *args, **kwargs):
        raise AssertionError('Unexpected image read during search')


class _Knowledge:
    async def read_propositions(self, *args, **kwargs):
        return []

    async def read_arguments(self, *args, **kwargs):
        return []


class _LearnerObservationSearch:
    def __init__(self, hits=()):
        self.hits = list(hits)
        self.calls = []

    async def search(self, user_uuid, query):
        self.calls.append((user_uuid, query))
        return self.hits


class _LearnerObservations:
    def __init__(self, records=()):
        self.records = (
            records
            if isinstance(records, dict)
            else {record.observation.uuid: record for record in records}
        )
        self.calls = []

    async def read(self, user_uuid, uuids):
        self.calls.append((user_uuid, uuids))
        return [self.records[uuid] for uuid in uuids]


def _tools(
    source_search,
    knowledge_search,
    evidence=None,
    *,
    learner_observation_search=None,
    learner_observations=None,
):
    source_evidence = evidence or _SourceEvidence()
    return TeachingEvidenceTools(
        source_search,
        source_evidence,
        SourceWorkspaceService(source_evidence),
        knowledge_search,
        _Knowledge(),
        learner_observation_search=(
            learner_observation_search or _LearnerObservationSearch()
        ),
        learner_observations=learner_observations or _LearnerObservations(),
        image_budget=ImagePayloadBudget(4, 26_214_400, 40_000_000),
        discovery_preview_characters=5,
        catalog_page_size=1,
    )


def _run_graph(node, request):
    return asyncio.run(node.run(request))


def test_local_acquisition_searches_and_registers_separate_ranked_results():
    source_a = Source(key='source-a')
    source_b = Source(key='source-b')
    block_a = SourceBlock(block_type=BlockType.PARAGRAPH, content='first')
    block_b = SourceBlock(block_type=BlockType.PARAGRAPH, content='second')
    source_search = _SourceSearch(
        [
            SourceSearchHit(
                source=source_a,
                block=block_a,
                page_indexes=[0],
                similarity=0.9,
                relevance_score=0.8,
            ),
            SourceSearchHit(
                source=source_b,
                block=block_b,
                page_indexes=[1],
                similarity=0.7,
                relevance_score=0.6,
            ),
        ]
    )
    proposition_a = Proposition(
        content='claim one',
        embedding=[1.0],
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    proposition_b = Proposition(
        content='claim two',
        embedding=[0.5],
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    knowledge_search = _KnowledgeSearch([proposition_a, proposition_b])
    tools = _tools(source_search, knowledge_search)
    graph = EvidenceAcquisitionNode(
        tools,
        source_search,
        knowledge_search,
        max_tool_calls=4,
    )
    workspace = TeachingEvidenceWorkspace()

    result = _run_graph(
        graph,
        EvidenceExecutionRequest(
            user_uuid='user-1',
            source_uuids=None,
            action=AcquireEvidenceAction(
                request=LocalEvidenceRequest(
                    question='Question', context='Need'
                )
            ),
            workspace=workspace,
        ),
    )
    observation = result.observation
    workspace = result.workspace

    assert [
        call[0] for call in source_search.calls + knowledge_search.calls
    ] == [
        'source',
        'proposition',
    ]
    assert source_search.calls[0][2] is None
    assert [
        workspace.block_registry[pos].block.uuid
        for pos in observation.block_positions
    ] == [block_a.uuid, block_b.uuid]
    assert [
        workspace.proposition_registry[pos].uuid
        for pos in observation.proposition_positions
    ] == [proposition_a.uuid, proposition_b.uuid]
    assert (
        workspace.local_outcomes[0].block_positions
        == observation.block_positions
    )
    assert (
        workspace.local_outcomes[0].proposition_positions
        == observation.proposition_positions
    )
    assert not workspace.receipts


def test_argument_discovery_tracks_latest_read_and_reuses_positions():
    now = datetime(2026, 10, 1, tzinfo=UTC)
    first = Proposition(
        uuid='first', content='First claim', embedding=[1.0], created_at=now
    )
    second = Proposition(
        uuid='second', content='Second claim', embedding=[1.0], created_at=now
    )
    records = {
        'first': PropositionRecord(
            proposition=first,
            assertions=[],
            argument_uuids=['old', 'shared'],
        ),
        'second': PropositionRecord(
            proposition=second,
            assertions=[],
            argument_uuids=['shared', 'new'],
        ),
    }

    class _ReadingKnowledge:
        async def read_propositions(self, user_uuid, uuids):
            return [records[uuid] for uuid in uuids]

        async def read_arguments(self, user_uuid, uuids):
            return [
                ArgumentRecord(
                    argument=Argument(
                        uuid=uuid,
                        explanation='Argument',
                        created_at=now,
                        embedding=[1.0, 0.0],
                    ),
                    premise_uuids=['first'],
                    conclusion_uuid='second',
                    supporting_block_uuids=[],
                )
                for uuid in uuids
            ]

    async def exercise():
        workspace = TeachingEvidenceWorkspace()
        first_pos = register_proposition(workspace, first)
        second_pos = register_proposition(workspace, second)
        source_evidence = _SourceEvidence()
        tools = TeachingEvidenceTools(
            _SourceSearch(),
            source_evidence,
            SourceWorkspaceService(source_evidence),
            _KnowledgeSearch(),
            _ReadingKnowledge(),
            learner_observation_search=_LearnerObservationSearch(),
            learner_observations=_LearnerObservations(),
            image_budget=ImagePayloadBudget(4, 26_214_400, 40_000_000),
            discovery_preview_characters=5,
            catalog_page_size=1,
        )

        async def read(*positions):
            return await tools.read_propositions(
                'user-1',
                None,
                ReadPropositionsAction(
                    kind='read_propositions',
                    proposition_positions=list(positions),
                ),
                workspace,
            )

        await read(first_pos)
        assert workspace.discovered_argument_positions == [0, 1]
        result = await read(second_pos)
        assert result.argument_positions == [1, 2]
        assert workspace.discovered_argument_positions == [1, 2]
        await read(first_pos, second_pos)
        assert workspace.discovered_argument_positions == [0, 1, 2]
        await read(second_pos)
        await tools.read_arguments(
            'user-1',
            None,
            ReadArgumentsAction(kind='read_arguments', argument_positions=[1]),
            workspace,
        )
        assert workspace.discovered_argument_positions == [0, 1, 2]
        await read(first_pos)
        assert workspace.discovered_argument_positions == [0, 1]
        assert [entry.uuid for entry in workspace.argument_registry] == [
            'old',
            'shared',
            'new',
        ]

    asyncio.run(exercise())


def test_argument_search_registers_preview_without_inspection():
    now = datetime(2026, 10, 1, tzinfo=UTC)
    argument = Argument(
        uuid='argument-1',
        explanation='The premises entail the conclusion.',
        embedding=[0.5, 0.25],
        created_at=now,
    )
    hit = ArgumentSearchHit(
        record=ArgumentRecord(
            argument=argument,
            premise_uuids=['premise-1'],
            conclusion_uuid='conclusion-1',
            supporting_block_uuids=[],
        ),
        content='Premise: one. Conclusion: two. Explanation: because.',
        similarity=0.9,
        relevance_score=0.8,
    )
    knowledge_search = _KnowledgeSearch(arguments=[hit])
    source_search = _SourceSearch()
    tools = _tools(source_search, knowledge_search)
    graph = EvidenceAcquisitionNode(
        tools, source_search, knowledge_search, max_tool_calls=4
    )
    workspace = TeachingEvidenceWorkspace()

    result = _run_graph(
        graph,
        EvidenceExecutionRequest(
            user_uuid='user-1',
            source_uuids=None,
            action=SearchArgumentsAction(
                kind='search_arguments', query='query'
            ),
            workspace=workspace,
        ),
    )

    assert result.observation.kind == 'search_arguments'
    assert result.observation.argument_positions == [0]
    assert result.workspace.discovered_argument_positions == [0]
    assert result.workspace.argument_registry[0].uuid == 'argument-1'
    assert result.workspace.argument_registry[0].preview == hit.content
    assert result.workspace.argument_registry[0].record is None
    assert result.workspace.active_argument_positions == []
    assert result.workspace.inspected_argument_positions == set()
    assert result.workspace.receipts == []
    assert knowledge_search.calls == [('argument', 'query')]


def test_learner_observation_search_discovery_then_explicit_read():
    now = datetime(2026, 10, 1, tzinfo=UTC)
    record = LearnerObservationRecord(
        observation=LearnerObservation(
            uuid='observation-1',
            kind=LearnerObservationKind.GAP,
            topic='vectors',
            description='Confuses direction with magnitude.',
            embedding=[0.5, 0.25],
            created_at=now,
        ),
        evidence=[
            LearnerObservationEvidence(
                turn_uuid='turn-1',
                session_uuid='session-1',
                content='The learner answer is private until read.',
                created_at=now,
                question=None,
            )
        ],
    )
    search = _LearnerObservationSearch(
        [
            LearnerObservationSearchHit(
                record=record, similarity=0.9, relevance_score=0.8
            )
        ]
    )
    observations = _LearnerObservations([record])
    source_search = _SourceSearch()
    knowledge_search = _KnowledgeSearch()
    tools = _tools(
        source_search,
        knowledge_search,
        learner_observation_search=search,
        learner_observations=observations,
    )
    graph = EvidenceAcquisitionNode(
        tools, source_search, knowledge_search, max_tool_calls=4
    )

    searched = _run_graph(
        graph,
        EvidenceExecutionRequest(
            user_uuid='user-1',
            source_uuids=None,
            action=SearchLearnerObservationsAction(
                kind='search_learner_observations', query='vectors'
            ),
            workspace=TeachingEvidenceWorkspace(),
        ),
    )
    assert searched.observation.learner_observation_positions == [0]
    assert searched.workspace.discovered_learner_observation_positions == [0]
    assert (
        searched.workspace.learner_observation_registry[0].observation.uuid
        == 'observation-1'
    )
    assert (
        searched.workspace.learner_observation_registry[0].observation.kind
        is LearnerObservationKind.GAP
    )
    assert (
        searched.workspace.learner_observation_registry[0].observation.topic
        == 'vectors'
    )
    assert (
        searched.workspace.learner_observation_registry[
            0
        ].observation.description
        == 'Confuses direction with magnitude.'
    )
    assert searched.workspace.learner_observation_registry[0].record is None
    assert searched.workspace.active_learner_observation_positions == []
    assert 'The learner answer' not in str(searched.workspace)

    read = _run_graph(
        graph,
        EvidenceExecutionRequest(
            user_uuid='user-1',
            source_uuids=None,
            action=ReadLearnerObservationsAction(
                kind='read_learner_observations',
                observation_positions=[0],
            ),
            workspace=searched.workspace,
        ),
    )
    assert read.observation.learner_observation_positions == [0]
    assert read.workspace.active_learner_observation_positions == [0]
    assert read.workspace.learner_observation_registry[0].record == record
    assert observations.calls == [('user-1', ['observation-1'])]

    cleared = _run_graph(
        graph,
        EvidenceExecutionRequest(
            user_uuid='user-1',
            source_uuids=None,
            action=ReadLearnerObservationsAction(
                kind='read_learner_observations', observation_positions=[]
            ),
            workspace=read.workspace,
        ),
    )
    assert cleared.observation.learner_observation_positions == []
    assert cleared.workspace.active_learner_observation_positions == []
    assert observations.calls == [('user-1', ['observation-1'])]


@pytest.mark.parametrize('include_hit', [False, True])
def test_statement_search_returns_typed_discovery_for_empty_and_selected_results(
    include_hit,
):
    source = Source(uuid='statement-source', key='linear algebra')
    statement = SourceStatement(
        uuid='statement-1',
        member_block_uuids=['block-1'],
        is_exercise=True,
        content='Determine whether v is an eigenvector.',
        embedding=[0.5, 0.25],
    )
    hits = (
        [
            SourceStatementSearchHit(
                statement=statement,
                source=source,
                page_indexes=[3],
                similarity=0.9,
                relevance_score=0.8,
            )
        ]
        if include_hit
        else []
    )

    class StatementSearch(_SourceSearch):
        def __init__(self):
            super().__init__()
            self.statement_calls = []

        async def search_statements(
            self, user_uuid, query, *, source_uuids=None, is_exercise=None
        ):
            self.statement_calls.append(
                (user_uuid, query, source_uuids, is_exercise)
            )
            return hits

    source_search = StatementSearch()
    knowledge_search = _KnowledgeSearch()
    graph = EvidenceAcquisitionNode(
        _tools(source_search, knowledge_search),
        source_search,
        knowledge_search,
        max_tool_calls=4,
    )
    result = _run_graph(
        graph,
        EvidenceExecutionRequest(
            user_uuid='user-1',
            source_uuids=None,
            action=SearchStatementsAction(
                kind='search_statements',
                query='eigenvector exercise',
                is_exercise=True,
            ),
            workspace=TeachingEvidenceWorkspace(),
        ),
    )

    expected_positions = [0] if include_hit else []
    assert isinstance(result.observation, TeachingToolObservation)
    assert result.observation.kind == 'search_statements'
    assert result.observation.statement_positions == expected_positions
    assert result.workspace.discovered_statement_positions == expected_positions
    assert result.workspace.active_statement_positions == []
    assert [
        entry.statement.uuid for entry in result.workspace.statement_registry
    ] == (['statement-1'] if include_hit else [])
    assert source_search.statement_calls == [
        ('user-1', 'eigenvector exercise', None, True)
    ]
