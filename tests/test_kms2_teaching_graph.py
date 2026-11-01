"""Behavioral tests for bounded teaching workflow graphs and commit mapping."""

import asyncio
from datetime import UTC, datetime

import pytest

from kms2.core.media_budget import ImagePayloadBudget
from kms2.core.model.block import SourceBlock
from kms2.core.model.block_types import BlockType
from kms2.core.model.knowledge.admission import KnowledgeAdmission
from kms2.core.model.page import SourcePage
from kms2.core.model.source import Source, SourceRevisionReference
from kms2.core.model.source_processing.admission import StagedSourceSnapshot
from kms2.core.model.teaching.inference import (
    RespondAction,
    SearchSourcesAction,
)
from kms2.core.model.teaching.observation import (
    LearnerObservation,
    LearnerObservationDraft,
    LearnerObservationKind,
    LearnerObservationWrite,
    NewLearnerObservationReference,
)
from kms2.core.model.teaching.operation import (
    CompleteProbing,
    ContinueProbing,
    TeachingAssessmentDraft,
    TeachingQuestionDraft,
)
from kms2.core.model.teaching.question import (
    FreeResponseSpecification,
    TeachingQuestionKind,
)
from kms2.core.model.teaching.session import (
    TeachingPhase,
    TeachingSession,
    TeachingSessionSnapshot,
    TeachingStatus,
)
from kms2.core.model.teaching.workflow import (
    TeachingInteractionRequest,
    TeachingInteractionResult,
    TeachingTurnRequest,
)
from kms2.core.model.teaching.workspace import (
    BlockInspection,
    TeachingEvidenceWorkspace,
)
from kms2.core.windowing import TokenBudget
from kms2.langgraph.evidence_acquisition.graph import EvidenceAcquisitionGraph
from kms2.langgraph.teaching.graph import TeachingGraph
from kms2.langgraph.teaching.interaction_graph import TeachingInteractionGraph
from kms2.langgraph.teaching.state import TeachingTurnState
from kms2.node.evidence_acquisition.acquisition import EvidenceAcquisitionNode
from kms2.node.teaching.evidence import staged_source_passages
from kms2.node.teaching.history import TeachingHistoryNode
from kms2.node.teaching.interaction import TeachingInteractionNode
from kms2.node.teaching.interaction_tools import TeachingInteractionToolsNode
from kms2.node.teaching.knowledge import TeachingKnowledgeNode
from kms2.node.teaching.observation import TeachingObservationNode
from kms2.node.teaching.operations import prepare_instructional_commit
from kms2.node.teaching.persistence import TeachingPersistenceNode, _commit
from kms2.node.teaching.teacher import TeachingTeacherNode
from kms2.node.teaching.tools import TeachingEvidenceTools
from kms2.service.learner_observations import LearnerObservationService
from kms2.service.teaching import TeachingService


class _Counter:
    def count_texts(self, texts):
        return [len(text) for text in texts]


class _Teacher:
    def __init__(self, actions):
        self.actions = list(actions)
        self.contexts = []

    async def aforward(self, *, context, evidence_images):
        self.contexts.append((context, evidence_images))
        return self.actions.pop(0)


class _SourceSearch:
    def __init__(self):
        self.calls = []

    async def search_sources(self, user_uuid, query, *, source_uuids=None):
        self.calls.append((user_uuid, query, source_uuids))
        return []

    async def search_statements(self, *args, **kwargs):
        return []

    async def search_procedures(self, *args, **kwargs):
        return []


class _KnowledgeSearch:
    async def search_propositions(self, user_uuid, query):
        return []


class _SourceEvidence:
    async def read_passages(self, *args, **kwargs):
        return []

    async def read_images_at_revisions(self, *args, **kwargs):
        return []


class _Knowledge:
    async def read_propositions(self, *args, **kwargs):
        return []

    async def read_reasoning_steps(self, *args, **kwargs):
        return []


def _snapshot(*, phase=TeachingPhase.PROBING, revision=4):
    now = datetime(2026, 1, 1, tzinfo=UTC)
    return TeachingSessionSnapshot(
        session=TeachingSession(
            uuid='session-1',
            goal='Understand linear transformations',
            status=TeachingStatus.AWAITING_TEACHER,
            phase=phase,
            revision=revision,
            created_at=now,
            updated_at=now,
        ),
        turns=[],
        current_assessment=None,
        current_plan=None,
        current_plan_acceptance=None,
        current_step_uuid=None,
        pending_question_uuid=None,
    )


def _response(
    operation,
    *,
    content='Response',
    content_kind='coordination',
    **kwargs,
):
    return RespondAction(
        content=content,
        content_kind=content_kind,
        operation=operation,
        **kwargs,
    )


def _interaction_graph(module, search, *, max_tool_calls):
    tools = TeachingEvidenceTools(
        search,
        _SourceEvidence(),
        _KnowledgeSearch(),
        _Knowledge(),
        discovery_preview_characters=40,
        catalog_page_size=4,
    )
    evidence_graph = EvidenceAcquisitionGraph(
        EvidenceAcquisitionNode(
            tools,
            search,
            _KnowledgeSearch(),
            researcher=None,
            source_admission=None,
            discovery_preview_characters=40,
        )
    )
    node = TeachingTeacherNode(
        module,
        TokenBudget(_Counter(), token_limit=20_000),
        max_tool_calls,
        image_budget=ImagePayloadBudget(
            max_images=4, max_bytes=1_000_000, max_pixels=1_000_000
        ),
        discovery_preview_characters=40,
        research_enabled=False,
    )
    return TeachingInteractionGraph(
        node,
        node,
        node,
        TeachingInteractionToolsNode(evidence_graph),
        max_tool_calls=max_tool_calls,
    )


def test_interaction_returns_to_phase_after_tool_and_enforces_limit():
    search = _SourceSearch()
    module = _Teacher(
        [
            SearchSourcesAction(kind='search_sources', query='local evidence'),
            _response(
                ContinueProbing(
                    kind='continue_probing',
                    question=TeachingQuestionDraft(
                        kind=TeachingQuestionKind.PROBE,
                        prompt='What does the map preserve?',
                        response=FreeResponseSpecification(),
                    ),
                )
            ),
        ]
    )
    graph = _interaction_graph(module, search, max_tool_calls=1)
    snapshot = _snapshot()
    request = TeachingInteractionRequest(
        turn=TeachingTurnRequest(
            user_uuid='user-1',
            session_uuid=snapshot.session.uuid,
            expected_revision=snapshot.session.revision,
            snapshot=snapshot,
            source_uuids=['source-1'],
        )
    )

    result = asyncio.run(graph.run(request))

    assert len(module.contexts) == 2
    assert module.contexts[0][0].phase is TeachingPhase.PROBING
    assert module.contexts[1][0].phase is TeachingPhase.PROBING
    assert search.calls == [('user-1', 'local evidence', ['source-1'])]
    assert result.response.operation.kind == 'continue_probing'
    assert [item.kind for item in result.workspace.observations] == [
        'search_sources'
    ]

    capped_search = _SourceSearch()
    capped = _interaction_graph(
        _Teacher(
            [
                SearchSourcesAction(
                    kind='search_sources',
                    query='must not run',
                )
            ]
        ),
        capped_search,
        max_tool_calls=0,
    )
    with pytest.raises(RuntimeError, match='tool-call limit'):
        asyncio.run(capped.run(request))
    assert capped_search.calls == []


def test_parent_completes_one_turn_after_more_tools_than_its_step_limit():
    source_search = _SourceSearch()
    teacher = _Teacher(
        [
            SearchSourcesAction(
                kind='search_sources', query=f'query-{position}'
            )
            for position in range(30)
        ]
        + [
            _response(
                ContinueProbing(
                    kind='continue_probing',
                    question=TeachingQuestionDraft(
                        kind=TeachingQuestionKind.PROBE,
                        prompt='What does the map preserve?',
                        response=FreeResponseSpecification(),
                    ),
                ),
                content='What does the map preserve?',
            )
        ]
    )
    interaction_graph = _interaction_graph(
        teacher, source_search, max_tool_calls=30
    )
    snapshot = _snapshot()
    final_view = TeachingService._view(snapshot)

    class _HistorySearch:
        async def search(self, user_uuid, query):
            return []

    class _TeachingService:
        def __init__(self):
            self.commits = []

        async def record_result(
            self,
            user_uuid,
            session_uuid,
            content,
            commit,
            *,
            expected_revision,
        ):
            self.commits.append(
                (user_uuid, session_uuid, content, commit, expected_revision)
            )
            return final_view

    teaching_service = _TeachingService()
    image_budget = ImagePayloadBudget(
        max_images=4, max_bytes=1_000_000, max_pixels=1_000_000
    )
    graph = TeachingGraph(
        TeachingHistoryNode(_HistorySearch()),
        TeachingInteractionNode(interaction_graph),
        TeachingObservationNode(LearnerObservationService(object()), object()),
        TeachingKnowledgeNode(object(), image_budget=image_budget),
        TeachingPersistenceNode(teaching_service),
    )
    request = TeachingTurnRequest(
        user_uuid='user-1',
        session_uuid=snapshot.session.uuid,
        expected_revision=snapshot.session.revision,
        snapshot=snapshot,
        source_uuids=['source-1'],
    )

    result = asyncio.run(graph.run(request))

    assert result == final_view
    assert len(teacher.contexts) == 31
    assert len(source_search.calls) == 30
    assert len(teaching_service.commits) == 1
    user_uuid, session_uuid, content, commit, expected_revision = (
        teaching_service.commits[0]
    )
    assert (user_uuid, session_uuid, content, expected_revision) == (
        'user-1',
        snapshot.session.uuid,
        'What does the map preserve?',
        snapshot.session.revision,
    )
    assert commit.phase is TeachingPhase.PROBING
    assert commit.new_question.prompt == 'What does the map preserve?'


def test_operation_and_persistence_map_observation_citation_source_and_knowledge():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    snapshot = _snapshot()
    source = Source(uuid='source-staged', revision=2, key='capture')
    block = SourceBlock(
        uuid='block-final', block_type=BlockType.PARAGRAPH, content='A claim.'
    )
    staged = StagedSourceSnapshot(
        source=source,
        pages=[SourcePage(index=0, markdown='A claim.', blocks=[block])],
    )
    observation = LearnerObservation(
        uuid='observation-1',
        kind=LearnerObservationKind.MISCONCEPTION,
        topic='linear maps',
        description='Thinks nonzero vectors cannot map to zero.',
        embedding=[0.1, 0.2],
        created_at=now,
    )
    prepared = LearnerObservationWrite(
        observation=observation, evidence_turn_uuids=['learner-turn']
    )
    assessment = TeachingAssessmentDraft(
        summary='The learner is developing the definition.',
        strands=[],
        observation_references=[
            NewLearnerObservationReference(kind='new', position=0)
        ],
    )
    response = _response(
        CompleteProbing(kind='complete_probing', assessment=assessment),
        content='A nonzero vector can map to zero.',
        content_kind='instruction',
        learner_observations=[
            LearnerObservationDraft(
                kind=LearnerObservationKind.MISCONCEPTION,
                topic='linear maps',
                description='Thinks nonzero vectors cannot map to zero.',
                evidence_turn_positions=[0],
            )
        ],
        cited_evidence_positions=[0],
    )
    workspace = TeachingEvidenceWorkspace(
        acquired_sources=[staged],
        staged_sources=[staged],
        receipts=[
            BlockInspection(source=source, block=block, page_indexes=[0])
        ],
    )
    support_source = Source(uuid='support-source', revision=7, key='support')
    knowledge = KnowledgeAdmission(
        expected_revision=3,
        source_revisions=[
            SourceRevisionReference(
                source_uuid=support_source.uuid,
                revision=support_source.revision,
            )
        ],
        new_propositions=[],
        proposition_uuids=[],
        assertions=[],
        new_reasoning_steps=[],
        existing_reasoning_step_uuids=[],
    )

    interaction = TeachingInteractionResult(
        response=response,
        workspace=workspace,
        visible_turn_indexes=[0],
        visible_observation_indexes=[],
        visible_citation_receipt_indexes=[0],
    )
    commit = prepare_instructional_commit(
        snapshot,
        response,
        visible_turn_indexes=[0],
        visible_observation_indexes=[],
        learner_history=[],
        prepared_observations=[prepared],
    )
    assert commit.phase is TeachingPhase.PLANNING
    assert commit.new_assessment.observation_uuids == ['observation-1']

    complete = _commit(
        snapshot,
        interaction,
        TeachingTurnState(
            request=TeachingTurnRequest(
                user_uuid='user-1',
                session_uuid=snapshot.session.uuid,
                expected_revision=snapshot.session.revision,
                snapshot=snapshot,
                source_uuids=None,
            ),
            interaction=interaction,
            prepared_observations=[prepared],
            knowledge_admission=knowledge,
        ),
    )

    assert [(item.kind, item.uuid) for item in complete.citations] == [
        ('block', 'block-final')
    ]
    assert complete.source_revisions == [
        SourceRevisionReference(source_uuid='source-staged', revision=2),
        SourceRevisionReference(source_uuid='support-source', revision=7),
    ]
    assert complete.staged_sources == [staged]
    assert complete.learner_observations == [prepared]
    assert complete.knowledge == knowledge


def test_staged_passage_selection_preserves_order_and_page_provenance():
    source = Source(key='staged')
    first = SourceBlock(block_type=BlockType.PARAGRAPH, content='first')
    second = SourceBlock(block_type=BlockType.PARAGRAPH, content='second')
    third = SourceBlock(block_type=BlockType.PARAGRAPH, content='third')
    staged = StagedSourceSnapshot(
        source=source,
        pages=[
            SourcePage(index=2, blocks=[first, second]),
            SourcePage(index=5, blocks=[third]),
        ],
    )

    passage = staged_source_passages(
        staged, [third.uuid, first.uuid], before=1, after=2
    )

    assert [item.block.uuid for item in passage.blocks] == [
        first.uuid,
        second.uuid,
        third.uuid,
    ]
    assert passage.selected_block_uuids == [third.uuid, first.uuid]
    assert [item.page_indexes for item in passage.blocks] == [[2], [2], [5]]
    assert passage.groups == []
