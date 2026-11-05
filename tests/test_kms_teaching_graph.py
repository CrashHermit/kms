"""Behavioral tests for bounded teaching workflow graphs and commit mapping."""

import asyncio
import hashlib
from datetime import UTC, datetime
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

from kms.core.media_budget import ImagePayloadBudget
from kms.core.model.block import SourceBlock
from kms.core.model.block_types import BlockType
from kms.core.model.knowledge.admission import KnowledgeAdmission
from kms.core.model.research.inference import (
    SubmitSourcesAction,
    WebSearchAction,
)
from kms.core.model.research.retrieval import WebSearchHit
from kms.core.model.source import Source, SourceRevisionReference
from kms.core.model.source_evidence import (
    SourceEvidenceBlock,
    SourceImage,
    SourcePassage,
    SourceProcedurePassage,
    SourceSearchHit,
    SourceStatementPassage,
)
from kms.core.model.source_processing.admission import CapturedPDFSource
from kms.core.model.source_semantic.source_procedure import SourceProcedure
from kms.core.model.source_semantic.source_statement import SourceStatement
from kms.core.model.source_workspace import (
    SourceWorkspaceBlockEntry,
    SourceWorkspaceImageEntry,
)
from kms.core.model.teaching.acquisition import (
    AcquireEvidenceAction,
    EvidenceResearchOutcome,
    ExternalEvidenceRequest,
    LocalEvidenceRequest,
    TeachingLocalOutcome,
)
from kms.core.model.teaching.inference import (
    FocusEvidenceAction,
    ReadImagesAction,
    ReadLearnerObservationsAction,
    ReadPassagesAction,
    ReadProceduresAction,
    ReadStatementsAction,
    RespondAction,
    SearchLearnerObservationsAction,
    SearchSourcesAction,
    TeachingArgumentDiscovery,
    TeachingContext,
    TeachingDiscoveryCatalog,
    TeachingImageDiscovery,
    TeachingLearnerObservation,
    TeachingToolObservation,
)
from kms.core.model.teaching.observation import (
    ExistingLearnerObservationReference,
    LearnerObservation,
    LearnerObservationDraft,
    LearnerObservationEvidence,
    LearnerObservationKind,
    LearnerObservationRecord,
    LearnerObservationSearchHit,
    LearnerObservationWrite,
    NewLearnerObservationReference,
)
from kms.core.model.teaching.operation import (
    Clarify,
    CompleteProbing,
    ContinueProbing,
    TeachingAssessmentDraft,
    TeachingQuestionDraft,
    TeachingResponseSegment,
)
from kms.core.model.teaching.question import (
    FreeResponseSpecification,
    TeachingQuestionKind,
)
from kms.core.model.teaching.session import (
    TeachingPhase,
    TeachingSession,
    TeachingSessionSnapshot,
    TeachingStatus,
)
from kms.core.model.teaching.workflow import (
    EvidenceExecutionResult,
    TeachingInteractionRequest,
    TeachingInteractionResult,
    TeachingTurnRequest,
)
from kms.core.model.teaching.workspace import (
    BlockInspection,
    ImageInspection,
    TeachingArgumentRegistryEntry,
    TeachingEvidenceWorkspace,
    TeachingLearnerObservationRegistryEntry,
    TeachingProcedureRegistryEntry,
    TeachingStatementRegistryEntry,
)
from kms.core.model.visual_asset import VisualAsset
from kms.core.tokenization.budget import InputBudgetExceeded, TokenBudget
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.langgraph.teaching.graph import TeachingGraph
from kms.langgraph.teaching.state import TeachingTurnState
from kms.node.evidence_acquisition.acquisition import EvidenceAcquisitionNode
from kms.node.research.researcher import ResearcherNode
from kms.node.research.tools import ResearchToolsNode
from kms.node.teaching.history import TeachingHistoryNode
from kms.node.teaching.interaction import TeachingInteractionNode
from kms.node.teaching.interaction_tools import TeachingInteractionToolsNode
from kms.node.teaching.knowledge import TeachingKnowledgeNode
from kms.node.teaching.observation import TeachingObservationNode
from kms.node.teaching.operations import prepare_instructional_commit
from kms.node.teaching.persistence import TeachingPersistenceNode, _commit
from kms.node.teaching.teacher import TeachingTeacherNode
from kms.node.teaching.tools import TeachingEvidenceTools
from kms.service.learner_observations import LearnerObservationService
from kms.service.source_workspace import SourceWorkspaceService
from kms.service.teaching import TeachingService


class _Counter:
    def count_texts(self, texts):
        return [len(text) for text in texts]


class _Teacher:
    def __init__(self, actions):
        self.actions = list(actions)
        self.contexts = []

    async def aforward(self, *, context, content_images):
        self.contexts.append((context, content_images))
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
        self.calls.append((user_uuid, list(uuids)))
        return [self.records[uuid] for uuid in uuids]


class _SourceEvidence:
    async def read_passages(self, *args, **kwargs):
        return []

    async def read_images_at_revisions(self, *args, **kwargs):
        return []


class _Knowledge:
    async def read_propositions(self, *args, **kwargs):
        return []

    async def read_arguments(self, *args, **kwargs):
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
    cited_evidence_positions=(),
    learner_observations=(),
):
    return RespondAction(
        segments=[
            TeachingResponseSegment(
                content=content,
                cited_evidence_positions=list(cited_evidence_positions),
            )
        ],
        content_kind=content_kind,
        learner_observations=list(learner_observations),
        operation=operation,
    )


def _interaction_graph(
    module,
    search,
    *,
    max_tool_calls,
    learner_observation_search=None,
    learner_observations=None,
):
    source_evidence = _SourceEvidence()
    observations = learner_observations or _LearnerObservations()
    tools = TeachingEvidenceTools(
        search,
        source_evidence,
        SourceWorkspaceService(source_evidence),
        _KnowledgeSearch(),
        _Knowledge(),
        learner_observation_search=(
            learner_observation_search or _LearnerObservationSearch()
        ),
        learner_observations=observations,
        image_budget=ImagePayloadBudget(4, 26_214_400, 40_000_000),
        discovery_preview_characters=40,
        catalog_page_size=4,
    )
    evidence = EvidenceAcquisitionNode(
        tools,
        search,
        _KnowledgeSearch(),
        max_tool_calls=0,
    )
    node = TeachingTeacherNode(
        module,
        TokenBudget(MaterialTokenEstimator(_Counter()), token_limit=20_000),
        max_tool_calls,
        image_budget=ImagePayloadBudget(
            max_images=4, max_bytes=1_000_000, max_pixels=1_000_000
        ),
        discovery_preview_characters=40,
        research_enabled=False,
    )

    class _NoOp:
        async def run(self, state):
            return {}

    return TeachingGraph(
        _NoOp(),
        _NoOp(),
        node,
        node,
        node,
        TeachingInteractionToolsNode(evidence),
        _NoOp(),
        _NoOp(),
        _NoOp(),
        None,
        None,
        evidence.postresearch,
        max_tool_calls=max_tool_calls,
        max_research_tool_calls=0,
    )


def _run_interaction(graph, request, workspace=None):
    state = asyncio.run(
        graph.run(
            TeachingTurnState(
                request=request.turn,
                workspace=workspace or TeachingEvidenceWorkspace(),
            )
        )
    )
    assert state.interaction is not None
    return state.interaction


def test_argument_discovery_exposes_only_bounded_search_previews():
    node = TeachingTeacherNode(
        _Teacher([]),
        TokenBudget(MaterialTokenEstimator(_Counter()), token_limit=100),
        0,
        image_budget=ImagePayloadBudget(
            max_images=4, max_bytes=1_000_000, max_pixels=1_000_000
        ),
        discovery_preview_characters=5,
        research_enabled=False,
    )
    workspace = TeachingEvidenceWorkspace(
        argument_registry=[
            TeachingArgumentRegistryEntry(uuid='searched', preview='abcdef'),
            TeachingArgumentRegistryEntry(uuid='linked'),
        ],
        discovered_argument_positions=[0, 1],
    )

    catalog = node._discovery(workspace)

    assert catalog.arguments == [
        TeachingArgumentDiscovery(position=0, preview='abcde'),
        TeachingArgumentDiscovery(position=1, preview=None),
    ]


def _interaction_action_state(
    action,
    *,
    visible_images=(),
    registry=(),
    learner_history=(),
    learner_registry=(),
):
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
    context = TeachingContext(
        goal='Learn',
        phase=TeachingPhase.PROBING,
        current_assessment=None,
        current_plan=None,
        current_plan_accepted=False,
        current_step_position=None,
        pending_question=None,
        learner_history=list(learner_history),
        images=[],
        observations=[],
        tools_remaining=1,
        history_truncated=False,
        discovery=TeachingDiscoveryCatalog(images=list(visible_images)),
    )
    return TeachingTurnState(
        request=request.turn,
        latest_action=action,
        latest_context=context,
        workspace=TeachingEvidenceWorkspace(
            image_registry=list(registry),
            learner_observation_registry=list(learner_registry),
        ),
    )


def test_learner_observation_read_rejects_unoffered_positions():
    state = _interaction_action_state(
        ReadLearnerObservationsAction(
            kind='read_learner_observations', observation_positions=[2]
        )
    )

    error = TeachingInteractionToolsNode.action_error(state)

    assert error is not None
    assert 'unavailable positions: [2]' in error


def test_learner_observation_read_accepts_a_visible_history_registry_handle():
    observation = LearnerObservation(
        uuid='visible-observation',
        kind=LearnerObservationKind.GAP,
        topic='vectors',
        description='Needs help with direction.',
        embedding=[0.1],
        created_at=datetime(2026, 10, 1, tzinfo=UTC),
    )
    learner_context = TeachingLearnerObservation(
        position=0,
        registry_position=0,
        kind=observation.kind,
        topic=observation.topic,
        description=observation.description,
        created_at=observation.created_at,
        evidence=[],
    )
    state = _interaction_action_state(
        ReadLearnerObservationsAction(
            kind='read_learner_observations', observation_positions=[0]
        ),
        learner_history=[learner_context],
        learner_registry=[
            TeachingLearnerObservationRegistryEntry(observation=observation)
        ],
    )

    assert TeachingInteractionToolsNode.action_error(state) is None


@pytest.mark.parametrize('kind', ['statement', 'procedure'])
def test_structured_source_read_registers_images_for_explicit_inspection(
    tmp_path, kind
):
    source = Source(uuid='structured-source', key='source')
    image_path = tmp_path / 'figure.png'
    image_path.write_bytes(_png('blue'))
    block = SourceBlock(
        uuid='structured-block',
        block_type=BlockType.IMAGE,
        content='Attached diagram.',
        assets=[
            VisualAsset(
                uuid='structured-image',
                path=str(image_path),
                content_sha256=hashlib.sha256(
                    image_path.read_bytes()
                ).hexdigest(),
            )
        ],
    )
    passage = SourcePassage(
        source=source,
        selected_block_uuids=[block.uuid],
        blocks=[SourceEvidenceBlock(block=block, page_indexes=[0])],
        groups=[],
    )
    statement = SourceStatement(
        uuid='statement',
        member_block_uuids=[block.uuid],
        content='Statement text.',
        embedding=[0.0],
    )
    procedure = SourceProcedure(
        uuid='procedure',
        member_block_uuids=[block.uuid],
        content='Procedure text.',
        embedding=[0.0],
    )

    class Evidence:
        async def read_statements(self, *_args):
            return [
                SourceStatementPassage(statement=statement, passages=[passage])
            ]

        async def read_procedures(self, *_args):
            return [
                SourceProcedurePassage(procedure=procedure, passages=[passage])
            ]

        async def read_images_at_revisions(self, user_uuid, references):
            assert user_uuid == 'user-1'
            assert references == [
                (source.uuid, source.revision, 'structured-image')
            ]
            return [
                SourceImage(asset=block.assets[0], data=image_path.read_bytes())
            ]

    workspace = TeachingEvidenceWorkspace()
    if kind == 'statement':
        workspace.statement_registry = [
            TeachingStatementRegistryEntry(
                source=source,
                page_indexes=[0],
                statement=statement,
            )
        ]
        action = ReadStatementsAction(
            kind='read_statements', statement_positions=[0]
        )
    else:
        workspace.procedure_registry = [
            TeachingProcedureRegistryEntry(
                source=source,
                page_indexes=[0],
                procedure=procedure,
            )
        ]
        action = ReadProceduresAction(
            kind='read_procedures', procedure_positions=[0]
        )
    source_evidence = Evidence()
    tools = TeachingEvidenceTools(
        None,
        source_evidence,
        SourceWorkspaceService(source_evidence),
        None,
        None,
        learner_observation_search=_LearnerObservationSearch(),
        learner_observations=_LearnerObservations(),
        image_budget=ImagePayloadBudget(1, 100_000, 100),
        discovery_preview_characters=100,
        catalog_page_size=10,
    )
    read = (
        tools.read_statements if kind == 'statement' else tools.read_procedures
    )

    observation = asyncio.run(read('user-1', None, action, workspace))

    assert observation.block_positions == [0]
    assert observation.image_positions == [0]
    assert workspace.active_block_positions == [0]
    assert workspace.active_image_positions == []
    assert workspace.discovered_image_positions == [0]
    assert workspace.read_image_positions == set()
    assert workspace.image_registry[0].data is None
    assert workspace.receipts == []
    inspected = asyncio.run(
        tools.read_images(
            'user-1',
            None,
            ReadImagesAction(kind='read_images', image_positions=[0]),
            workspace,
        )
    )
    assert inspected.image_positions == [0]
    assert workspace.active_image_positions == [0]
    assert workspace.image_registry[0].data == image_path.read_bytes()
    rejecting_tools = TeachingEvidenceTools(
        None,
        source_evidence,
        SourceWorkspaceService(source_evidence),
        None,
        None,
        learner_observation_search=_LearnerObservationSearch(),
        learner_observations=_LearnerObservations(),
        image_budget=ImagePayloadBudget(1, 1, 100),
        discovery_preview_characters=100,
        catalog_page_size=10,
    )
    rejected = asyncio.run(
        rejecting_tools.read_images(
            'user-1',
            None,
            ReadImagesAction(kind='read_images', image_positions=[0]),
            workspace,
        )
    )
    assert rejected.error is not None
    assert rejected.image_positions == []
    assert workspace.active_image_positions == [0]
    assert workspace.image_registry[0].data == image_path.read_bytes()
    state = _interaction_action_state(_response(Clarify(kind='clarify')))
    state.workspace = workspace
    module = _Teacher([_response(Clarify(kind='clarify'))])
    node = TeachingTeacherNode(
        module,
        TokenBudget(MaterialTokenEstimator(_Counter()), token_limit=100_000),
        0,
        image_budget=ImagePayloadBudget(
            max_images=1, max_bytes=100_000, max_pixels=100
        ),
        discovery_preview_characters=100,
        research_enabled=False,
    )
    asyncio.run(node.run(state))
    context, _ = module.contexts[0]
    assert [image.position for image in context.discovery.images] == [0]
    assert [(group.kind, group.material.text) for group in context.groups] == [
        (kind, f'{kind.capitalize()} text.')
    ]
    assert [receipt.kind for receipt in workspace.receipts] == [
        'block',
        'image',
    ]


def test_passage_read_discovers_images_without_loading_them(tmp_path):
    source = Source(uuid='passage-source', key='source')
    image_path = tmp_path / 'figure.png'
    image_path.write_bytes(_png('blue'))
    block = SourceBlock(
        uuid='passage-block',
        block_type=BlockType.IMAGE,
        content='Attached diagram.',
        assets=[VisualAsset(uuid='passage-image', path=str(image_path))],
    )
    passage = SourcePassage(
        source=source,
        selected_block_uuids=[block.uuid],
        blocks=[SourceEvidenceBlock(block=block, page_indexes=[0])],
        groups=[],
    )

    class Evidence:
        async def read_passages(self, *_args, **_kwargs):
            return [passage]

        async def read_images_at_revisions(self, *_args):
            raise AssertionError('Passage reading must not load image bytes')

    evidence = Evidence()
    workspace = TeachingEvidenceWorkspace(
        block_registry=[
            SourceWorkspaceBlockEntry(
                source=source,
                page_indexes=[0],
                block=block,
            )
        ]
    )
    tools = TeachingEvidenceTools(
        None,
        evidence,
        SourceWorkspaceService(evidence),
        None,
        None,
        learner_observation_search=_LearnerObservationSearch(),
        learner_observations=_LearnerObservations(),
        image_budget=ImagePayloadBudget(1, 100_000, 100),
        discovery_preview_characters=100,
        catalog_page_size=10,
    )

    observation = asyncio.run(
        tools.read_passages(
            'user-1',
            None,
            ReadPassagesAction(
                kind='read_passages',
                block_positions=[0],
                expand_memberships=False,
                before=0,
                after=0,
            ),
            workspace,
        )
    )

    assert observation.image_positions == [0]
    assert workspace.discovered_image_positions == [0]
    assert workspace.active_image_positions == []
    assert workspace.read_image_positions == set()
    assert workspace.image_registry[0].data is None


@pytest.mark.parametrize(
    'availability', ['read_image_positions', 'inspected_image_positions']
)
def test_focus_uses_only_explicitly_selected_available_images(availability):
    source = Source(uuid='focus-source', key='source')
    block = SourceBlock(block_type=BlockType.IMAGE, content='diagram')
    workspace = TeachingEvidenceWorkspace(
        block_registry=[
            SourceWorkspaceBlockEntry(
                source=source, page_indexes=[0], block=block
            )
        ],
        image_registry=[
            SourceWorkspaceImageEntry(
                asset=VisualAsset(uuid='read', path='read.png'),
                source=source,
                block_positions=[0],
                data=_png('green'),
            ),
            SourceWorkspaceImageEntry(
                asset=VisualAsset(uuid='also-read', path='also-read.png'),
                source=source,
                block_positions=[0],
                data=_png('red'),
            ),
            SourceWorkspaceImageEntry(
                asset=VisualAsset(uuid='unread', path='unread.png'),
                source=source,
                block_positions=[0],
            ),
        ],
        inspected_block_positions={0},
    )
    setattr(workspace, availability, {0, 1})
    tools = TeachingEvidenceTools(
        None,
        None,
        SourceWorkspaceService(None),
        None,
        None,
        learner_observation_search=_LearnerObservationSearch(),
        learner_observations=_LearnerObservations(),
        image_budget=ImagePayloadBudget(1, 100_000, 100),
        discovery_preview_characters=100,
        catalog_page_size=10,
    )

    focused = asyncio.run(
        tools.focus_evidence(
            'user-1',
            None,
            FocusEvidenceAction(kind='focus_evidence', block_positions=[0]),
            workspace,
        )
    )
    assert focused.image_positions == []
    assert workspace.active_image_positions == []
    selected = asyncio.run(
        tools.focus_evidence(
            'user-1',
            None,
            FocusEvidenceAction(
                kind='focus_evidence',
                block_positions=[0],
                image_positions=[0],
            ),
            workspace,
        )
    )
    assert selected.image_positions == [0]
    assert workspace.image_registry[1].data == _png('red')
    too_many = asyncio.run(
        tools.focus_evidence(
            'user-1',
            None,
            FocusEvidenceAction(
                kind='focus_evidence',
                image_positions=[0, 1],
            ),
            workspace,
        )
    )
    assert too_many.error is not None
    assert workspace.active_image_positions == [0]

    unread = asyncio.run(
        tools.focus_evidence(
            'user-1',
            None,
            FocusEvidenceAction(kind='focus_evidence', image_positions=[2]),
            workspace,
        )
    )
    assert unread.image_positions == []
    assert workspace.image_registry[2].data is None


def test_teacher_keeps_legacy_text_receipt_text_only():
    source = Source(uuid='legacy-source', key='source')
    block = SourceBlock(
        uuid='legacy-block', block_type=BlockType.IMAGE, content='caption'
    )
    image = SourceWorkspaceImageEntry(
        asset=VisualAsset(uuid='discovered-image', path='unread.png'),
        source=source,
        block_positions=[0],
    )
    state = _interaction_action_state(_response(Clarify(kind='clarify')))
    state.workspace.block_registry = [
        SourceWorkspaceBlockEntry(source=source, page_indexes=[0], block=block)
    ]
    state.workspace.inspected_block_positions = {0}
    state.workspace.active_block_positions = [0]
    state.workspace.image_registry = [image]
    state.workspace.discovered_image_positions = [0]

    module = _Teacher([_response(Clarify(kind='clarify'))])
    node = TeachingTeacherNode(
        module,
        TokenBudget(MaterialTokenEstimator(_Counter()), token_limit=10000),
        0,
        image_budget=ImagePayloadBudget(
            max_images=1, max_bytes=100_000, max_pixels=100
        ),
        discovery_preview_characters=100,
        research_enabled=False,
    )
    asyncio.run(node.run(state))

    context, content_images = module.contexts[0]
    assert context.material.blocks[0].material.text == 'caption'
    assert context.material.blocks[0].material.image_positions == []
    assert context.material.blocks[0].asset_positions == []
    assert context.discovery.images[0].position == 0
    assert content_images == []


def test_teacher_image_cost_gates_pinned_state_before_dispatch():
    source = Source(uuid='image-source', key='source')
    block = SourceBlock(
        uuid='image-block', block_type=BlockType.IMAGE, content=''
    )
    image_bytes = _png('blue')
    workspace = TeachingEvidenceWorkspace(
        block_registry=[
            SourceWorkspaceBlockEntry(
                source=source, page_indexes=[0], block=block
            )
        ],
        image_registry=[
            SourceWorkspaceImageEntry(
                asset=VisualAsset(uuid='image-asset', path='unused.png'),
                source=source,
                block_positions=[0],
                data=image_bytes,
            )
        ],
        active_block_positions=[0],
        inspected_block_positions={0},
        active_image_positions=[0],
        inspected_image_positions={0},
        read_image_positions={0},
    )
    state = _interaction_action_state(_response(Clarify(kind='clarify')))
    state.workspace = workspace
    budget_counter = MaterialTokenEstimator(_Counter())
    probe = TeachingTeacherNode(
        _Teacher([]),
        TokenBudget(budget_counter, 100_000),
        0,
        image_budget=ImagePayloadBudget(1, 100_000, 100),
        discovery_preview_characters=100,
        research_enabled=False,
    )
    context, images = asyncio.run(
        probe._context(
            state,
            [],
            [],
            history_truncated=False,
            learner_history_truncated=False,
        )
    )
    payload_tokens = budget_counter.estimate(
        text=context.model_dump_json(), images=images
    )
    assert len(images) == 1
    assert (
        payload_tokens
        == budget_counter.count_texts([context.model_dump_json()])[0] + 2
    )

    rejected_predictor = _Teacher([_response(Clarify(kind='clarify'))])
    rejected = TeachingTeacherNode(
        rejected_predictor,
        TokenBudget(budget_counter, payload_tokens - 1),
        0,
        image_budget=ImagePayloadBudget(1, 100_000, 100),
        discovery_preview_characters=100,
        research_enabled=False,
    )
    with pytest.raises(InputBudgetExceeded, match='Pinned teaching state'):
        asyncio.run(rejected.run(state))
    assert rejected_predictor.contexts == []

    accepted_predictor = _Teacher([_response(Clarify(kind='clarify'))])
    accepted = TeachingTeacherNode(
        accepted_predictor,
        TokenBudget(budget_counter, payload_tokens),
        0,
        image_budget=ImagePayloadBudget(1, 100_000, 100),
        discovery_preview_characters=100,
        research_enabled=False,
    )
    asyncio.run(accepted.run(state))
    assert len(accepted_predictor.contexts[0][1]) == 1


def test_interaction_rejects_unoffered_image_positions_without_dispatch():
    source = Source(uuid='source-1', key='source')
    registry_entry = SourceWorkspaceImageEntry(
        asset=VisualAsset(uuid='image-1', path='image.png'),
        source=source,
        block_positions=[0],
    )

    class _Evidence:
        def __init__(self):
            self.calls = []

        async def run(self, request):
            self.calls.append(request)
            raise AssertionError('Invalid image reference must not dispatch')

    for registry in ([], [registry_entry]):
        evidence = _Evidence()
        node = TeachingInteractionToolsNode(evidence)
        state = _interaction_action_state(
            ReadImagesAction(kind='read_images', image_positions=[0]),
            registry=registry,
        )
        result = asyncio.run(node.run(state))

        assert evidence.calls == []
        assert state.workspace.observations == []
        assert result['workspace'].model_dump(exclude={'observations'}) == (
            state.workspace.model_dump(exclude={'observations'})
        )
        assert result['tool_calls'] == 1
        observation = result['workspace'].observations[-1]
        assert observation.kind == 'invalid_action'
        assert 'image positions' in observation.error


def test_interaction_dispatches_offered_image_position():
    source = Source(uuid='source-1', key='source')
    registry_entry = SourceWorkspaceImageEntry(
        asset=VisualAsset(uuid='image-1', path='image.png'),
        source=source,
        block_positions=[0],
    )

    class _Evidence:
        def __init__(self):
            self.calls = []

        async def run(self, request):
            self.calls.append(request)
            return EvidenceExecutionResult(
                workspace=request.workspace,
                observation=TeachingToolObservation(
                    kind='read_images',
                    query=None,
                    block_positions=[0],
                    image_positions=[0],
                    groups=[],
                ),
            )

    evidence = _Evidence()
    node = TeachingInteractionToolsNode(evidence)
    state = _interaction_action_state(
        ReadImagesAction(kind='read_images', image_positions=[0]),
        visible_images=[
            TeachingImageDiscovery(position=0, block_positions=[0])
        ],
        registry=[registry_entry],
    )

    result = asyncio.run(node.run(state))

    assert len(evidence.calls) == 1
    assert result['tool_calls'] == 1
    assert result['workspace'].observations[-1].kind == 'read_images'


def test_interaction_rejects_external_research_for_restricted_sources():
    state = _interaction_action_state(
        AcquireEvidenceAction(
            request=ExternalEvidenceRequest(local_outcome_position=0)
        )
    )
    state.latest_context.research_enabled = True
    state.latest_context.local_outcomes = [
        TeachingLocalOutcome(
            position=0,
            question='Question',
            block_positions=[],
            proposition_positions=[],
            expanded=False,
            research=None,
        )
    ]

    class Evidence:
        async def run(self, state):
            raise AssertionError('Restricted external research was dispatched')

    result = asyncio.run(TeachingInteractionToolsNode(Evidence()).run(state))

    assert result['workspace'].observations[-1].kind == 'invalid_action'
    assert (
        'sources are restricted' in result['workspace'].observations[-1].error
    )


def test_source_search_scope_must_be_an_offered_research_source():
    source_revision = SourceRevisionReference(
        source_uuid='new-source', revision=3
    )
    state = _interaction_action_state(
        SearchSourcesAction(
            kind='search_sources',
            query='follow up',
            source_uuids=['new-source'],
        )
    )
    state.request.source_uuids = None
    state.latest_context.local_outcomes = [
        TeachingLocalOutcome(
            position=0,
            question='Question',
            block_positions=[],
            proposition_positions=[],
            expanded=True,
            research=EvidenceResearchOutcome(
                failures=[], source_revisions=[source_revision]
            ),
        )
    ]

    assert TeachingInteractionToolsNode.action_error(state) is None

    state.latest_action = SearchSourcesAction(
        kind='search_sources', query='other', source_uuids=['other-source']
    )
    assert 'unavailable sources' in TeachingInteractionToolsNode.action_error(
        state
    )


def test_locally_found_block_can_be_read_without_receipt():
    action = ReadPassagesAction(
        kind='read_passages',
        block_positions=[4],
        expand_memberships=False,
        before=0,
        after=0,
    )
    state = _interaction_action_state(action)
    state.latest_context.local_outcomes = [
        TeachingLocalOutcome(
            position=0,
            question='Find a diagram',
            block_positions=[4],
            proposition_positions=[],
            expanded=False,
            research=None,
        )
    ]

    class Evidence:
        async def run(self, request):
            assert request.action.block_positions == [4]
            assert request.workspace.receipts == []
            return EvidenceExecutionResult(
                workspace=request.workspace,
                observation=TeachingToolObservation(
                    kind='read_passages',
                    query=None,
                    block_positions=[4],
                    image_positions=[],
                    groups=[],
                ),
            )

    result = asyncio.run(TeachingInteractionToolsNode(Evidence()).run(state))
    assert result['workspace'].observations[-1].kind == 'read_passages'
    assert result['workspace'].receipts == []


def test_interaction_returns_rejected_action_observation_to_teacher():
    module = _Teacher(
        [
            ReadImagesAction(kind='read_images', image_positions=[0]),
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
    source_search = _SourceSearch()
    graph = _interaction_graph(module, source_search, max_tool_calls=1)
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

    result = _run_interaction(graph, request)

    assert len(module.contexts) == 2
    assert module.contexts[1][0].observations[0].kind == 'invalid_action'
    assert 'image positions' in module.contexts[1][0].observations[0].error
    assert source_search.calls == []
    assert result.workspace.image_registry == []
    assert [item.kind for item in result.workspace.observations] == [
        'invalid_action'
    ]


def test_interaction_rejects_response_citations_outside_visible_candidates():
    operation = ContinueProbing(
        kind='continue_probing',
        question=TeachingQuestionDraft(
            kind=TeachingQuestionKind.PROBE,
            prompt='What does the map preserve?',
            response=FreeResponseSpecification(),
        ),
    )
    module = _Teacher(
        [
            _response(operation, cited_evidence_positions=[0]),
            _response(operation),
        ]
    )
    graph = _interaction_graph(module, _SourceSearch(), max_tool_calls=1)
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

    result = _run_interaction(graph, request)

    assert len(module.contexts) == 2
    observation = module.contexts[1][0].observations[0]
    assert observation.kind == 'invalid_action'
    assert 'citation_candidates' in observation.error
    assert result.response.segments[0].cited_evidence_positions == []
    assert [item.kind for item in result.workspace.observations] == [
        'invalid_action'
    ]


def test_invalid_response_at_tool_cap_reports_reference_error():
    operation = ContinueProbing(
        kind='continue_probing',
        question=TeachingQuestionDraft(
            kind=TeachingQuestionKind.PROBE,
            prompt='What does the map preserve?',
            response=FreeResponseSpecification(),
        ),
    )
    graph = _interaction_graph(
        _Teacher([_response(operation, cited_evidence_positions=[0])]),
        _SourceSearch(),
        max_tool_calls=0,
    )
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
    with pytest.raises(
        RuntimeError, match='invalid references at the tool-call limit'
    ):
        _run_interaction(graph, request)


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

    result = _run_interaction(graph, request)

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
        _run_interaction(capped, request)
    assert capped_search.calls == []


def test_parent_completes_one_turn_after_more_tools_than_its_step_limit():
    source_search = _SourceSearch()
    history_record = LearnerObservationRecord(
        observation=LearnerObservation(
            uuid='automatic-observation',
            kind=LearnerObservationKind.PARTIAL_UNDERSTANDING,
            topic='maps',
            description='Understands images but not embeddings.',
            embedding=[0.2],
            created_at=datetime(2026, 10, 1, tzinfo=UTC),
        ),
        evidence=[],
    )
    history_search = _LearnerObservationSearch(
        [
            LearnerObservationSearchHit(
                record=history_record, similarity=0.9, relevance_score=0.8
            )
        ]
    )
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
    snapshot = _snapshot()
    final_view = TeachingService._view(snapshot)

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
    learner_observations = LearnerObservationService(object())
    source_evidence = _SourceEvidence()
    knowledge_search = _KnowledgeSearch()
    evidence_tools = TeachingEvidenceTools(
        source_search,
        source_evidence,
        SourceWorkspaceService(source_evidence),
        knowledge_search,
        _Knowledge(),
        learner_observation_search=_LearnerObservationSearch(),
        learner_observations=_LearnerObservations(),
        image_budget=image_budget,
        discovery_preview_characters=40,
        catalog_page_size=4,
    )
    evidence_node = EvidenceAcquisitionNode(
        evidence_tools, source_search, knowledge_search, max_tool_calls=0
    )
    teacher_node = TeachingTeacherNode(
        teacher,
        TokenBudget(MaterialTokenEstimator(_Counter()), token_limit=20_000),
        30,
        image_budget=image_budget,
        discovery_preview_characters=40,
        research_enabled=False,
    )
    graph = TeachingGraph(
        TeachingHistoryNode(history_search),
        TeachingInteractionNode(source_evidence, learner_observations),
        teacher_node,
        teacher_node,
        teacher_node,
        TeachingInteractionToolsNode(evidence_node),
        TeachingObservationNode(learner_observations, object()),
        TeachingKnowledgeNode(object(), image_budget=image_budget),
        TeachingPersistenceNode(teaching_service),
        None,
        None,
        evidence_node.postresearch,
        max_tool_calls=30,
        max_research_tool_calls=0,
    )
    request = TeachingTurnRequest(
        user_uuid='user-1',
        session_uuid=snapshot.session.uuid,
        expected_revision=snapshot.session.revision,
        snapshot=snapshot,
        source_uuids=['source-1'],
    )

    result_state = asyncio.run(graph.run(TeachingTurnState(request=request)))

    assert result_state.final_view == final_view
    assert len(teacher.contexts) == 31
    assert history_search.calls == [
        ('user-1', 'Understand linear transformations')
    ]
    assert (
        teacher.contexts[0][0].learner_history[0].description
        == 'Understands images but not embeddings.'
    )
    assert teacher.contexts[0][0].learner_history[0].registry_position == 0
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


def test_flat_graph_runs_two_independent_research_budgets_and_scoped_retrieval():
    class ScopedSearch:
        def __init__(self):
            self.calls = []

        async def search_sources(self, user_uuid, query, *, source_uuids=None):
            self.calls.append((query, source_uuids))
            if source_uuids is None:
                return []
            source_uuid = source_uuids[0]
            return [
                SourceSearchHit(
                    source=Source(uuid=source_uuid, key=source_uuid),
                    block=SourceBlock(
                        uuid=f'{source_uuid}-block',
                        block_type=BlockType.PARAGRAPH,
                        content='Admitted evidence',
                    ),
                    page_indexes=[0],
                    similarity=0.9,
                    relevance_score=0.9,
                )
            ]

        async def search_statements(self, *args, **kwargs):
            return []

        async def search_procedures(self, *args, **kwargs):
            return []

    class KnowledgeSearch:
        async def search_propositions(self, user_uuid, query):
            return []

        async def search_arguments(self, user_uuid, query):
            return []

    class Researcher:
        def __init__(self):
            self.actions = [
                action
                for _ in range(2)
                for action in (
                    WebSearchAction(kind='web_search', query='find source'),
                    SubmitSourcesAction(
                        kind='submit_sources', result_positions=[0]
                    ),
                )
            ]
            self.contexts = []

        async def aforward(self, *, context):
            self.contexts.append(context)
            return self.actions.pop(0)

    class Web:
        def __init__(self):
            self.queries = []

        async def search(self, query):
            self.queries.append(query)
            index = len(self.queries)
            return [
                WebSearchHit(
                    title=f'Source {index}',
                    url=f'https://example.test/{index}',
                    snippet='discovery text',
                )
            ]

    class Capture:
        async def capture_source(self, url, *, title=None):
            return CapturedPDFSource(
                url=url,
                title=title,
                retrieved_at=datetime(2026, 10, 1, tzinfo=UTC),
                data=b'pdf',
            )

    class Admission:
        def __init__(self):
            self.sources = []

        async def admit(self, user_uuid, capture):
            source = Source(
                uuid=f'admitted-{len(self.sources) + 1}', key=capture.url
            )
            self.sources.append(source)
            return SimpleNamespace(source=source)

    class NoOp:
        async def run(self, state):
            return {}

    source_search = ScopedSearch()
    knowledge_search = KnowledgeSearch()
    source_evidence = _SourceEvidence()
    evidence_tools = TeachingEvidenceTools(
        source_search,
        source_evidence,
        SourceWorkspaceService(source_evidence),
        knowledge_search,
        _Knowledge(),
        learner_observation_search=_LearnerObservationSearch(),
        learner_observations=_LearnerObservations(),
        image_budget=ImagePayloadBudget(4, 1_000_000, 1_000_000),
        discovery_preview_characters=40,
        catalog_page_size=4,
    )
    evidence = EvidenceAcquisitionNode(
        evidence_tools, source_search, knowledge_search, max_tool_calls=2
    )
    teacher = _Teacher(
        [
            AcquireEvidenceAction(
                request=LocalEvidenceRequest(
                    question='First question', context='First context'
                )
            ),
            AcquireEvidenceAction(
                request=ExternalEvidenceRequest(local_outcome_position=0)
            ),
            AcquireEvidenceAction(
                request=LocalEvidenceRequest(
                    question='Second question', context='Second context'
                )
            ),
            AcquireEvidenceAction(
                request=ExternalEvidenceRequest(local_outcome_position=1)
            ),
            _response(
                ContinueProbing(
                    kind='continue_probing',
                    question=TeachingQuestionDraft(
                        kind=TeachingQuestionKind.PROBE,
                        prompt='What is supported?',
                        response=FreeResponseSpecification(),
                    ),
                ),
                content='What is supported?',
            ),
        ]
    )
    image_budget = ImagePayloadBudget(4, 1_000_000, 1_000_000)
    token_budget = TokenBudget(
        MaterialTokenEstimator(_Counter()), token_limit=20_000
    )
    teacher_node = TeachingTeacherNode(
        teacher,
        token_budget,
        4,
        image_budget=image_budget,
        discovery_preview_characters=40,
        research_enabled=True,
    )
    research_module = Researcher()
    researcher = ResearcherNode(research_module, token_budget)
    admission = Admission()
    research_tools = ResearchToolsNode(Capture(), Web(), admission)
    graph = TeachingGraph(
        TeachingHistoryNode(_LearnerObservationSearch()),
        TeachingInteractionNode(source_evidence, _LearnerObservations()),
        teacher_node,
        teacher_node,
        teacher_node,
        TeachingInteractionToolsNode(evidence),
        NoOp(),
        NoOp(),
        NoOp(),
        researcher,
        research_tools,
        evidence.postresearch,
        max_tool_calls=4,
        max_research_tool_calls=2,
    )
    request = TeachingTurnRequest(
        user_uuid='user-1',
        session_uuid='session-1',
        expected_revision=0,
        snapshot=_snapshot(),
        source_uuids=None,
    )

    state = asyncio.run(graph.run(TeachingTurnState(request=request)))

    assert state.interaction is not None
    assert state.tool_calls == 4
    assert [context.task.question for context in research_module.contexts] == [
        'First question',
        'First question',
        'Second question',
        'Second question',
    ]
    assert [call[1] for call in source_search.calls] == [
        None,
        ['admitted-1'],
        None,
        ['admitted-2'],
    ]
    assert len(admission.sources) == 2
    assert [
        outcome.research.source_revisions[0].source_uuid
        for outcome in state.interaction.workspace.local_outcomes
    ] == ['admitted-1', 'admitted-2']


def test_operation_and_persistence_map_observation_citation_source_and_knowledge():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    snapshot = _snapshot()
    source = Source(uuid='source-referenced', revision=2, key='capture')
    block = SourceBlock(
        uuid='block-final', block_type=BlockType.PARAGRAPH, content='A claim.'
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
    unused_source = Source(uuid='unused-source', revision=5, key='unused')
    unused_block = SourceBlock(
        uuid='unused-block',
        block_type=BlockType.PARAGRAPH,
        content='Uncited search hit.',
    )
    workspace = TeachingEvidenceWorkspace(
        block_registry=[
            SourceWorkspaceBlockEntry(
                source=unused_source,
                page_indexes=[4],
                block=unused_block,
            )
        ],
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
        new_arguments=[],
        existing_argument_uuids=[],
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
        learner_observation_uuids=[],
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
        SourceRevisionReference(source_uuid='source-referenced', revision=2),
        SourceRevisionReference(source_uuid='support-source', revision=7),
    ]
    assert complete.learner_observations == [prepared]
    assert complete.knowledge == knowledge


def _png(color: str) -> bytes:
    image = Image.new('RGB', (2, 2), color)
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()


class _KnowledgeGraphCapture:
    def __init__(self):
        self.calls = []

    async def run(self, state):
        self.calls.append(state)
        return state


def _knowledge_state(workspace):
    receipt_positions = list(range(len(workspace.receipts)))
    return SimpleNamespace(
        request=SimpleNamespace(user_uuid='user-1'),
        interaction=SimpleNamespace(
            response=RespondAction(
                segments=[
                    TeachingResponseSegment(
                        content='The response teaches a claim.',
                        cited_evidence_positions=receipt_positions,
                    )
                ],
                content_kind='instruction',
                operation=Clarify(kind='clarify'),
            ),
            workspace=workspace,
            visible_citation_receipt_indexes=receipt_positions,
        ),
    )


def test_knowledge_evidence_places_only_inspected_images_in_nested_context():
    source = Source(uuid='knowledge-source', key='source')
    block = SourceBlock(
        uuid='knowledge-block',
        block_type=BlockType.PARAGRAPH,
        content='Inspected text.',
    )
    workspace = TeachingEvidenceWorkspace(
        block_registry=[
            SourceWorkspaceBlockEntry(
                source=source,
                page_indexes=[0],
                block=block,
            )
        ],
        inspected_block_positions={0},
        active_block_positions=[0],
        image_registry=[
            SourceWorkspaceImageEntry(
                asset=VisualAsset(uuid='unread-image', path='unread.png'),
                source=source,
                block_positions=[0],
                data=_png('red'),
            )
        ],
        receipts=[
            BlockInspection(source=source, block=block, page_indexes=[0])
        ],
    )
    graph = _KnowledgeGraphCapture()
    node = TeachingKnowledgeNode(
        graph,
        image_budget=ImagePayloadBudget(
            max_images=1, max_bytes=100_000, max_pixels=100
        ),
    )

    asyncio.run(node.run(_knowledge_state(workspace)))

    request = graph.calls[0]
    assert (
        request.request.blocks[0].source_block.material.text
        == 'Inspected text.'
    )
    assert request.request.blocks[0].source_block.material.image_positions == []
    assert request.content_image_bytes == []


def test_knowledge_evidence_maps_only_authorized_images_and_budgets_before_graph():
    source = Source(uuid='knowledge-source', key='source')
    block = SourceBlock(
        uuid='knowledge-block',
        block_type=BlockType.IMAGE,
        content='Uninspected text.',
    )
    selected_image = _png('blue')
    workspace = TeachingEvidenceWorkspace(
        block_registry=[
            SourceWorkspaceBlockEntry(
                source=source,
                page_indexes=[0],
                block=block,
            )
        ],
        inspected_image_positions={1, 2},
        active_image_positions=[1, 2],
        image_registry=[
            SourceWorkspaceImageEntry(
                asset=VisualAsset(uuid='unread-image', path='unread.png'),
                source=source,
                block_positions=[0],
                data=_png('red'),
            ),
            SourceWorkspaceImageEntry(
                asset=VisualAsset(uuid='read-image', path='read.png'),
                source=source,
                block_positions=[0],
                data=selected_image,
            ),
            SourceWorkspaceImageEntry(
                asset=VisualAsset(uuid='read-sibling', path='read-sibling.png'),
                source=source,
                block_positions=[0],
                data=_png('green'),
            ),
        ],
        receipts=[
            ImageInspection(
                source=source,
                asset=VisualAsset(uuid='read-image', path='read.png'),
                block_uuids=[block.uuid],
                data=selected_image,
            ),
            ImageInspection(
                source=source,
                asset=VisualAsset(uuid='read-sibling', path='read-sibling.png'),
                block_uuids=[block.uuid],
                data=_png('green'),
            ),
        ],
    )

    graph = _KnowledgeGraphCapture()
    state = _knowledge_state(workspace)
    state.interaction.response.segments = [
        TeachingResponseSegment(
            content='Image-supported passage one.',
            cited_evidence_positions=[0],
        ),
        TeachingResponseSegment(
            content='Image-supported passage two.',
            cited_evidence_positions=[1],
        ),
    ]
    asyncio.run(
        TeachingKnowledgeNode(
            graph,
            image_budget=ImagePayloadBudget(
                max_images=2, max_bytes=100_000, max_pixels=100
            ),
        ).run(state)
    )

    request = graph.calls[0]
    assert request.request.blocks[0].source_block.material.text is None
    assert request.request.blocks[0].source_block.material.image_positions == [
        0,
        1,
    ]
    assert [
        segment.image_positions for segment in request.request.segments
    ] == [
        [0],
        [1],
    ]
    assert request.content_image_bytes == [
        selected_image,
        _png('green'),
    ]

    rejected_graph = _KnowledgeGraphCapture()
    with pytest.raises(InputBudgetExceeded, match='Active image evidence'):
        asyncio.run(
            TeachingKnowledgeNode(
                rejected_graph,
                image_budget=ImagePayloadBudget(
                    max_images=1, max_bytes=100_000, max_pixels=100
                ),
            ).run(state)
        )
    assert rejected_graph.calls == []


def test_learner_observation_read_uses_compact_positions_and_registry_uuids():
    now = datetime(2026, 10, 1, tzinfo=UTC)
    record = LearnerObservationRecord(
        observation=LearnerObservation(
            uuid='searched-observation',
            kind=LearnerObservationKind.GAP,
            topic='vectors',
            description='Confuses direction with magnitude.',
            embedding=[0.5],
            created_at=now,
        ),
        evidence=[
            LearnerObservationEvidence(
                turn_uuid='learner-turn',
                session_uuid='session-1',
                content='The learner said that direction equals magnitude.',
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
    operation = CompleteProbing(
        kind='complete_probing',
        assessment=TeachingAssessmentDraft(
            summary='The learner confuses two vector properties.',
            strands=[],
            observation_references=[
                ExistingLearnerObservationReference(kind='existing', position=0)
            ],
        ),
    )
    teacher = _Teacher(
        [
            SearchLearnerObservationsAction(
                kind='search_learner_observations', query='vectors'
            ),
            ReadLearnerObservationsAction(
                kind='read_learner_observations', observation_positions=[1]
            ),
            _response(operation),
        ]
    )
    graph = _interaction_graph(
        teacher,
        _SourceSearch(),
        max_tool_calls=2,
        learner_observation_search=search,
        learner_observations=observations,
    )
    snapshot = _snapshot()
    request = TeachingInteractionRequest(
        turn=TeachingTurnRequest(
            user_uuid='user-1',
            session_uuid=snapshot.session.uuid,
            expected_revision=snapshot.session.revision,
            snapshot=snapshot,
            source_uuids=None,
        )
    )
    prior_observation = LearnerObservation(
        uuid='prior-unread',
        kind=LearnerObservationKind.GAP,
        topic='matrices',
        description='Unrelated old gap.',
        embedding=[0.1],
        created_at=now,
    )
    workspace = TeachingEvidenceWorkspace(
        learner_observation_registry=[
            TeachingLearnerObservationRegistryEntry(
                observation=prior_observation
            )
        ]
    )

    result = _run_interaction(graph, request, workspace)

    assert search.calls == [('user-1', 'vectors')]
    assert observations.calls == [('user-1', ['searched-observation'])]
    assert teacher.contexts[0][0].learner_history == []
    assert (
        teacher.contexts[1][0].discovery.learner_observations[0].position == 1
    )
    assert (
        teacher.contexts[1][0].discovery.learner_observations[0].kind
        is LearnerObservationKind.GAP
    )
    assert (
        teacher.contexts[1][0].discovery.learner_observations[0].topic
        == 'vectors'
    )
    assert (
        teacher.contexts[1][0].discovery.learner_observations[0].preview
        == 'Confuses direction with magnitude.'
    )
    assert teacher.contexts[1][0].learner_history == []
    assert teacher.contexts[2][0].learner_history[0].position == 0
    assert teacher.contexts[2][0].learner_history[0].registry_position == 1
    assert (
        teacher.contexts[2][0].learner_history[0].evidence[0].content
        == 'The learner said that direction equals magnitude.'
    )
    assert result.visible_observation_indexes == [1]

    commit = prepare_instructional_commit(
        snapshot,
        result.response,
        visible_turn_indexes=result.visible_turn_indexes,
        visible_observation_indexes=result.visible_observation_indexes,
        learner_observation_uuids=[
            entry.observation.uuid
            for entry in result.workspace.learner_observation_registry
        ],
        prepared_observations=[],
    )
    assert commit.new_assessment.observation_uuids == ['searched-observation']
