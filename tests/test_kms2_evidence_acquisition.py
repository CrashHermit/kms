"""Behavioral tests for isolated evidence acquisition and inspection."""

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

from kms2.core.model.block import SourceBlock
from kms2.core.model.block_types import BlockType
from kms2.core.model.knowledge.proposition import (
    Proposition,
    PropositionRecord,
    ReasoningStep,
)
from kms2.core.model.knowledge.reasoning import ReasoningStepRecord
from kms2.core.model.page import SourcePage
from kms2.core.model.source import Source
from kms2.core.model.source_evidence import SourceSearchHit
from kms2.core.model.source_processing.admission import StagedSourceSnapshot
from kms2.core.model.teaching.acquisition import (
    AcquireEvidenceAction,
    ExternalEvidenceRequest,
    LocalEvidenceOutcome,
    LocalEvidenceRequest,
)
from kms2.core.model.teaching.inference import (
    FocusEvidenceAction,
    ListAcquiredBlocksAction,
    ReadImagesAction,
    ReadPassagesAction,
    ReadPropositionsAction,
    ReadReasoningStepsAction,
)
from kms2.core.model.teaching.workflow import EvidenceExecutionRequest
from kms2.core.model.teaching.workspace import TeachingEvidenceWorkspace
from kms2.langgraph.evidence_acquisition.graph import EvidenceAcquisitionGraph
from kms2.node.evidence_acquisition.acquisition import EvidenceAcquisitionNode
from kms2.node.teaching.evidence import (
    active_block_positions,
    register_block,
    register_image,
    register_proposition,
)
from kms2.node.teaching.tools import TeachingEvidenceTools


class _SourceSearch:
    def __init__(self, hits=()):
        self.hits = list(hits)
        self.calls = []

    async def search_sources(self, user_uuid, query, *, source_uuids=None):
        self.calls.append(('source', query, source_uuids))
        return self.hits

    async def search_statements(self, *args, **kwargs):
        return []

    async def search_procedures(self, *args, **kwargs):
        return []


class _KnowledgeSearch:
    def __init__(self, propositions=()):
        self.propositions = list(propositions)
        self.calls = []

    async def search_propositions(self, user_uuid, query):
        self.calls.append(('proposition', query))
        return [SimpleNamespace(proposition=item) for item in self.propositions]


class _SourceEvidence:
    def __init__(self):
        self.owned_reads = []

    async def read_passages(self, *args, **kwargs):
        self.owned_reads.append(args)
        raise AssertionError(
            'staged evidence must not read through the database'
        )

    async def read_images_at_revisions(self, *args, **kwargs):
        raise AssertionError('staged images must not read through the database')


class _Knowledge:
    async def read_propositions(self, *args, **kwargs):
        return []

    async def read_reasoning_steps(self, *args, **kwargs):
        return []


class _Researcher:
    def __init__(self):
        self.task = None

    async def run(self, task):
        self.task = task
        return SimpleNamespace(
            brief='Orientation only.',
            failures=[],
            source_materials=[object()],
        )


class _Admission:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.calls = 0

    async def run(self, user_uuid, material):
        self.calls += 1
        return self.snapshot


def _tools(source_search, knowledge_search, evidence=None):
    return TeachingEvidenceTools(
        source_search,
        evidence or _SourceEvidence(),
        knowledge_search,
        _Knowledge(),
        discovery_preview_characters=5,
        catalog_page_size=1,
    )


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
    graph = EvidenceAcquisitionGraph(
        EvidenceAcquisitionNode(
            tools,
            source_search,
            knowledge_search,
            researcher=None,
            source_admission=None,
            discovery_preview_characters=5,
        )
    )
    workspace = TeachingEvidenceWorkspace()

    result = asyncio.run(
        graph.run(
            EvidenceExecutionRequest(
                user_uuid='user-1',
                source_uuids=['source-a'],
                action=AcquireEvidenceAction(
                    request=LocalEvidenceRequest(
                        question='Question', context='Need', urls=[]
                    )
                ),
                workspace=workspace,
            )
        )
    )
    observation = result.observation
    workspace = result.workspace

    assert [
        call[0] for call in source_search.calls + knowledge_search.calls
    ] == [
        'source',
        'proposition',
    ]
    assert source_search.calls[0][2] == ['source-a']
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


def test_external_acquisition_stages_without_inspection_and_paging_is_bounded(
    tmp_path,
):
    image_path = tmp_path / 'image.png'
    image_path.write_bytes(b'png-bytes')
    source = Source(key='captured')
    image_block = SourceBlock(
        block_type=BlockType.IMAGE,
        content='figure text',
        assets=[{'path': str(image_path)}],
    )
    other_block = SourceBlock(block_type=BlockType.PARAGRAPH, content='later')
    snapshot = StagedSourceSnapshot(
        source=source,
        pages=[
            SourcePage(
                index=0, markdown='capture', blocks=[image_block, other_block]
            )
        ],
    )
    researcher = _Researcher()
    admission = _Admission(snapshot)
    source_search = _SourceSearch()
    knowledge_search = _KnowledgeSearch()
    tools = _tools(source_search, knowledge_search)
    node = EvidenceAcquisitionNode(
        tools,
        source_search,
        knowledge_search,
        researcher,
        admission,
        discovery_preview_characters=5,
    )
    graph = EvidenceAcquisitionGraph(node)
    workspace = TeachingEvidenceWorkspace()
    local_source = Source(key='local')
    local_block = SourceBlock(
        block_type=BlockType.PARAGRAPH, content='secret local full text'
    )
    local_position = register_block(workspace, local_source, local_block, [0])
    inspected_block = SourceBlock(
        block_type=BlockType.PARAGRAPH, content='complete inspected evidence'
    )
    inspected_position = register_block(
        workspace, local_source, inspected_block, [1]
    )
    workspace.inspected_block_positions.add(inspected_position)
    workspace.active_block_positions.append(inspected_position)
    workspace.local_outcomes.append(
        LocalEvidenceOutcome(
            request=LocalEvidenceRequest(
                question='Need', context='Context', urls=[]
            ),
            block_positions=[local_position, inspected_position],
            proposition_positions=[],
        )
    )

    result = asyncio.run(
        graph.run(
            EvidenceExecutionRequest(
                user_uuid='user-1',
                source_uuids=None,
                action=AcquireEvidenceAction(
                    request=ExternalEvidenceRequest(local_outcome_position=0)
                ),
                workspace=workspace,
            )
        )
    )

    assert researcher.task.context == 'Context'
    assert researcher.task.local_findings[0].text == 'secre'
    assert researcher.task.local_findings[0].inspected is False
    assert (
        researcher.task.local_findings[1].text == 'complete inspected evidence'
    )
    assert researcher.task.local_findings[1].inspected is True
    assert admission.calls == 1
    assert len(result.workspace.staged_sources) == 1
    assert len(result.workspace.acquired_sources) == 1
    assert not result.workspace.receipts
    assert (
        result.workspace.local_outcomes[0].research.brief == 'Orientation only.'
    )
    assert result.observation.acquired_source_positions == [0]
    assert [
        result.workspace.block_registry[position].block.uuid
        for position in result.workspace.discovered_block_positions
    ] == [image_block.uuid]

    page = asyncio.run(
        tools.list_acquired_blocks(
            'user-1',
            None,
            ListAcquiredBlocksAction(source_position=0, start=1),
            result.workspace,
        )
    )
    assert len(page.block_positions) == 1
    assert page.catalog_start == 1
    assert not result.workspace.receipts


def test_staged_passage_and_image_reads_create_only_successful_receipts(
    tmp_path,
):
    image_path = tmp_path / 'asset.png'
    image_path.write_bytes(b'image-content')
    source = Source(key='staged')
    text = SourceBlock(block_type=BlockType.PARAGRAPH, content='neighbor')
    image = SourceBlock(
        block_type=BlockType.IMAGE,
        content='figure',
        assets=[{'path': str(image_path)}],
    )
    snapshot = StagedSourceSnapshot(
        source=source, pages=[SourcePage(index=3, blocks=[text, image])]
    )
    workspace = TeachingEvidenceWorkspace(
        staged_sources=[snapshot], acquired_sources=[snapshot]
    )
    source_search = _SourceSearch()
    tools = _tools(source_search, _KnowledgeSearch())
    text_pos = register_block(workspace, source, text, [3])
    image_pos = register_block(workspace, source, image, [3])
    asset_pos = register_image(workspace, image.assets[0], source, image_pos)

    read = asyncio.run(
        tools.read_passages(
            'user-1',
            None,
            ReadPassagesAction(
                kind='read_passages',
                block_positions=[image_pos],
                expand_memberships=True,
                before=1,
                after=0,
            ),
            workspace,
        )
    )
    assert [
        workspace.block_registry[pos].block.uuid for pos in read.block_positions
    ] == [text.uuid, image.uuid]
    assert active_block_positions(workspace) == [text_pos, image_pos]
    asyncio.run(
        tools.focus_evidence(
            'user-1',
            None,
            FocusEvidenceAction(
                block_positions=[], image_positions=[asset_pos]
            ),
            workspace,
        )
    )
    image_read = asyncio.run(
        tools.read_images(
            'user-1',
            None,
            ReadImagesAction(kind='read_images', image_positions=[asset_pos]),
            workspace,
        )
    )
    assert image_read.image_positions == [asset_pos]
    assert active_block_positions(workspace) == []
    assert len(workspace.receipts) == 3
    assert workspace.receipts[0].kind == 'block'
    assert workspace.receipts[0].block.uuid == text.uuid
    assert workspace.image_registry[asset_pos].data == b'image-content'


def test_reasoning_discovery_tracks_latest_read_and_reuses_positions():
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
            reasoning_step_uuids=['old', 'shared'],
        ),
        'second': PropositionRecord(
            proposition=second,
            assertions=[],
            reasoning_step_uuids=['shared', 'new'],
        ),
    }

    class _ReadingKnowledge:
        async def read_propositions(self, user_uuid, uuids):
            return [records[uuid] for uuid in uuids]

        async def read_reasoning_steps(self, user_uuid, uuids):
            return [
                ReasoningStepRecord(
                    step=ReasoningStep(
                        uuid=uuid, explanation='Argument', created_at=now
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
        tools = TeachingEvidenceTools(
            _SourceSearch(),
            _SourceEvidence(),
            _KnowledgeSearch(),
            _ReadingKnowledge(),
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
        assert workspace.discovered_reasoning_positions == [0, 1]
        result = await read(second_pos)
        assert result.reasoning_positions == [1, 2]
        assert workspace.discovered_reasoning_positions == [1, 2]
        await read(first_pos, second_pos)
        assert workspace.discovered_reasoning_positions == [0, 1, 2]
        await read(second_pos)
        await tools.read_reasoning_steps(
            'user-1',
            None,
            ReadReasoningStepsAction(
                kind='read_reasoning_steps', reasoning_positions=[1]
            ),
            workspace,
        )
        assert workspace.discovered_reasoning_positions == [0, 1, 2]
        await read(first_pos)
        assert workspace.discovered_reasoning_positions == [0, 1]
        assert [entry.uuid for entry in workspace.reasoning_registry] == [
            'old',
            'shared',
            'new',
        ]

    asyncio.run(exercise())
