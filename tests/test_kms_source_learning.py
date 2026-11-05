import asyncio
from contextlib import asynccontextmanager
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

import kms.node.source_learning.atomic as atomic_node
import kms.service.content as content_service
from kms import application, tui
from kms.config.inference import ContextWindowSettings
from kms.config.settings import Settings
from kms.core.content import MultimodalContentBuilder
from kms.core.media_budget import ImagePayloadBudget
from kms.core.model.block import SourceBlock
from kms.core.model.block_types import BlockType
from kms.core.model.context import SourceBlockWindow
from kms.core.model.source import Source
from kms.core.model.source_learning.atomic import (
    SourceAtomicFlashcardInput,
    SourceAtomicFlashcardMaterial,
    SourceAtomicFlashcardRequest,
)
from kms.core.model.source_learning.coherent import (
    SourceAtomicFlashcardPacket,
    SourceCoherentFlashcardCandidate,
    SourceCoherentFlashcardRequest,
)
from kms.core.model.source_learning.flashcard import (
    SourceFlashcard,
    SourceFlashcardCandidate,
    SourceFlashcardOccurrence,
)
from kms.core.model.visual_asset import VisualAsset
from kms.core.tokenization.budget import InputBudgetExceeded, TokenBudget
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.database.source_learning.repository import SourceLearningRepository
from kms.langgraph.source_learning.graph import SourceLearningGraph
from kms.langgraph.source_learning.state import SourceLearningState
from kms.node.source_learning.atomic import SourceAtomicFlashcardNode
from kms.node.source_learning.atomic_persistence import (
    SourceAtomicFlashcardPersistenceNode,
)
from kms.node.source_learning.cleanup import SourceLearningCleanupNode
from kms.node.source_learning.coherent import SourceCoherentFlashcardNode
from kms.node.source_learning.coherent_persistence import (
    SourceCoherentFlashcardPersistenceNode,
)


def test_source_learning_application_rejects_unowned_source(monkeypatch):
    async def list_sources(settings, user_uuid):
        return []

    monkeypatch.setattr(application, 'list_sources', list_sources)

    with pytest.raises(ValueError, match='not owned by user'):
        asyncio.run(
            application.run_source_learning_stage(
                Settings(), 'user-1', 'foreign-source'
            )
        )


def test_tui_exposes_owned_source_learning_action(monkeypatch):
    from kms.core.model.user import User

    user = User(uuid='user-1', name='Alex')
    source = Source(uuid='source-1', key='book.pdf')
    prompts = iter([tui.SOURCE_LEARNING_OPTION, source, True])
    selections = []
    calls = []

    class Prompt:
        def execute(self):
            return next(prompts)

    async def list_users(settings):
        return [user]

    async def list_sources(settings, user_uuid):
        return [source]

    async def list_unowned_sources(settings):
        return []

    async def run_learning(settings, user_uuid, source_uuid):
        calls.append((user_uuid, source_uuid))
        return SimpleNamespace(
            atomic_flashcard_count=2,
            coherent_flashcard_count=2,
        )

    monkeypatch.setattr(tui, 'list_users', list_users)
    monkeypatch.setattr(tui, 'list_sources', list_sources)
    monkeypatch.setattr(tui, 'list_unowned_sources', list_unowned_sources)
    monkeypatch.setattr(tui, 'run_source_learning_stage', run_learning)
    monkeypatch.setattr(
        tui,
        'inquirer',
        SimpleNamespace(
            select=lambda **kwargs: selections.append(kwargs) or Prompt(),
            confirm=lambda **kwargs: Prompt(),
        ),
    )

    tui._run_tui()

    assert tui.SOURCE_LEARNING_OPTION in selections[0]['choices']
    assert tui.EXISTING_SOURCE_OPTION in selections[0]['choices']
    assert calls == [('user-1', 'source-1')]


def test_source_learning_application_runs_owned_source(monkeypatch):
    source = Source(uuid='source-1', key='book.pdf')
    resources = SimpleNamespace(
        local_models=object(),
        tokenizers=object(),
        database=object(),
        recorder=None,
    )
    calls = []

    async def list_sources(settings, user_uuid):
        return [source]

    class Graph:
        async def run(self, initial_state):
            calls.append(initial_state)
            return initial_state.model_copy(
                update={
                    'atomic_flashcard_count': 2,
                    'coherent_flashcard_count': 4,
                }
            )

    @asynccontextmanager
    async def application_resources(
        settings, *, embeddings_required, reranking_required=False
    ):
        assert embeddings_required is False
        assert reranking_required is False
        yield resources

    monkeypatch.setattr(application, 'list_sources', list_sources)
    monkeypatch.setattr(
        application, '_application_resources', application_resources
    )
    monkeypatch.setattr(
        application,
        'build_source_learning_graph',
        lambda *args, **kwargs: Graph(),
    )

    result = asyncio.run(
        application.run_source_learning_stage(Settings(), 'user-1', source.uuid)
    )

    assert result.atomic_flashcard_count == 2
    assert result.coherent_flashcard_count == 4
    assert calls == [SourceLearningState(source_uuid=source.uuid)]


class _Counter:
    def count_texts(self, texts):
        return [1 if text else 0 for text in texts]


class _Repository:
    def __init__(self, requests):
        self.requests = requests

    async def load_atomic_flashcard_materials(self, source_uuid):
        return [
            SourceAtomicFlashcardMaterial(
                source_block_uuid=request.source_block_uuid,
                source_page_index=request.source_page_index,
            )
            for request in self.requests
        ]


class _BlockRepository:
    def __init__(self, blocks):
        self.blocks = blocks

    async def load_blocks(self, source_uuid):
        return self.blocks


class _AtomicModule:
    def __init__(self):
        self.calls = []

    async def acall(
        self, *, request: SourceAtomicFlashcardInput, content_images
    ):
        self.calls.append((request, content_images))
        return [
            SourceFlashcardCandidate(
                question=f'Question: {request.evidence.source_text}',
                answer=request.evidence.source_text or 'Visual material.',
            )
        ]


class _CoherentModule:
    def __init__(self, card_indexes=(1, 2)):
        self.inputs = []
        self.images = []
        self.card_indexes = card_indexes

    async def acall(self, *, request, content_images):
        self.inputs.append(request)
        self.images.append(content_images)
        return [
            SourceCoherentFlashcardCandidate(
                question='How are the claims connected?',
                answer='They share the supplied source context.',
                card_indexes=list(self.card_indexes),
            )
        ]


def _request(block_key, page_index=0):
    block = SourceBlock(
        uuid=f'block-{block_key}',
        block_type=BlockType.PARAGRAPH,
        content='The source defines the average velocity.',
    )
    return SourceAtomicFlashcardRequest(
        source_block_uuid=block.uuid,
        source_page_index=page_index,
        window=SourceBlockWindow(target=[block]),
    )


def test_two_pass_generation_retains_context_and_uses_local_indexes():
    requests = [
        _request('1'),
        _request('2'),
        _request('3'),
    ]
    repository = _Repository(requests)
    budget = TokenBudget(
        counter=MaterialTokenEstimator(_Counter()), token_limit=100
    )
    atomic = SourceAtomicFlashcardNode(
        repository,
        _AtomicModule(),
        ContextWindowSettings(backward_budget=1, forward_budget=1),
        _BlockRepository(
            [
                SourceBlock(
                    uuid=request.source_block_uuid,
                    block_type=BlockType.PARAGRAPH,
                    content='The source defines the average velocity.',
                )
                for request in requests
            ]
        ),
        budget=budget,
        image_budget=ImagePayloadBudget(4, 26214400, 40000000),
    )
    state = SourceLearningState(
        source_uuid='source-1', atomic_requests=requests
    )
    sends = asyncio.run(atomic.dispatch(state))
    assert len(sends) == 3
    first = asyncio.run(atomic.worker(sends[0].arg))
    second = asyncio.run(atomic.worker(sends[1].arg))
    third = asyncio.run(atomic.worker(sends[2].arg))
    atomic_state = state.model_copy(
        update={
            'atomic_results': [
                *third['atomic_results'],
                *second['atomic_results'],
                *first['atomic_results'],
            ]
        }
    )
    packet_state = atomic.collect(atomic_state)
    coherent_module = _CoherentModule(card_indexes=(1, 3))
    coherent = SourceCoherentFlashcardNode(
        coherent_module,
        budget=budget,
        image_budget=ImagePayloadBudget(4, 26214400, 40000000),
    )
    coherent_state = atomic_state.model_copy(update=packet_state)
    prepared = asyncio.run(coherent.prepare(coherent_state))
    request = prepared['coherent_requests'][0]
    model_input = request.model_input(MultimodalContentBuilder())
    serialized = model_input.model_dump_json()
    assert 'hub-1' not in serialized
    assert 'source-1' not in serialized
    assert [card.index for card in model_input.packets[0].cards] == [1]
    assert [card.index for card in model_input.packets[1].cards] == [2]
    assert [card.index for card in model_input.packets[2].cards] == [3]
    assert [
        block.material.text
        for block in model_input.packets[
            0
        ].evidence.source_context.target_blocks
    ] == ['The source defines the average velocity.']
    coherent_state = coherent_state.model_copy(update=prepared)
    result = asyncio.run(coherent.worker({'ordinal': 0, 'request': request}))
    collected = coherent.collect(
        coherent_state.model_copy(
            update={'coherent_results': result['coherent_results']}
        )
    )
    occurrence: SourceFlashcardOccurrence = collected['coherent_occurrences'][0]
    assert occurrence.source_block_uuids == [
        'block-1',
        'block-3',
    ]
    assert len(occurrence.derived_card_uuids) == 2
    assert occurrence.derived_card_uuids == [
        coherent_state.atomic_packets[0].cards[0].card.uuid,
        coherent_state.atomic_packets[2].cards[0].card.uuid,
    ]
    assert coherent_module.inputs[0].packets[0].evidence.source_text


def test_atomic_load_assembles_independent_token_bounded_context_windows():
    requests = [_request('1'), _request('2')]
    requests[1].source_block_uuid = 'block-3'
    blocks = [
        SourceBlock(
            uuid=f'block-{index}',
            block_type=BlockType.PARAGRAPH,
            content=content,
        )
        for index, content in enumerate(
            [
                'before first',
                'first owner',
                'between owners',
                'second owner',
                'after second',
            ]
        )
    ]
    atomic = SourceAtomicFlashcardNode(
        _Repository(requests),
        _AtomicModule(),
        ContextWindowSettings(backward_budget=1, forward_budget=1),
        _BlockRepository(blocks),
        budget=TokenBudget(
            counter=MaterialTokenEstimator(_Counter()), token_limit=100
        ),
        image_budget=ImagePayloadBudget(4, 26214400, 40000000),
    )

    loaded = asyncio.run(
        atomic.load(SourceLearningState(source_uuid='source-1'))
    )['atomic_requests']

    first_context = loaded[0].window
    second_context = loaded[1].window
    assert [block.content for block in first_context.context_before] == [
        'before first'
    ]
    assert [block.content for block in first_context.target] == ['first owner']
    assert [block.content for block in first_context.context_after] == [
        'between owners'
    ]
    assert [block.content for block in second_context.context_before] == [
        'between owners'
    ]
    assert [block.content for block in second_context.target] == [
        'second owner'
    ]
    assert [block.content for block in second_context.context_after] == [
        'after second'
    ]
    for request in loaded:
        assert (
            request.source_block_uuid
            not in request.model_input(
                MultimodalContentBuilder()
            ).model_dump_json()
        )


@pytest.mark.parametrize('has_evidence', [True, False])
def test_learning_graph_persists_selected_provenance_after_atomic_parents(
    has_evidence,
):
    requests = (
        [
            _request('1'),
            _request('2'),
        ]
        if has_evidence
        else []
    )

    class Repository(_Repository):
        def __init__(self):
            super().__init__(requests)
            self.cards = {'old-card': None}

        async def clear_source_learning(self, source_uuid):
            self.cards.clear()

        async def persist_flashcards(self, source_uuid, occurrences):
            for occurrence in occurrences:
                for parent_uuid in occurrence.derived_card_uuids:
                    assert parent_uuid in self.cards
                self.cards[occurrence.card.uuid] = occurrence
            return len(occurrences)

    repository = Repository()
    budget = TokenBudget(
        counter=MaterialTokenEstimator(_Counter()), token_limit=100
    )
    graph = SourceLearningGraph(
        SourceLearningCleanupNode(repository),
        SourceAtomicFlashcardNode(
            repository,
            _AtomicModule(),
            ContextWindowSettings(backward_budget=1, forward_budget=1),
            _BlockRepository(
                [
                    SourceBlock(
                        uuid=request.source_block_uuid,
                        block_type=BlockType.PARAGRAPH,
                        content='The source defines the average velocity.',
                    )
                    for request in requests
                ]
            ),
            budget=budget,
            image_budget=ImagePayloadBudget(4, 26214400, 40000000),
        ),
        SourceAtomicFlashcardPersistenceNode(repository),
        SourceCoherentFlashcardNode(
            _CoherentModule(),
            budget=budget,
            image_budget=ImagePayloadBudget(4, 26214400, 40000000),
        ),
        SourceCoherentFlashcardPersistenceNode(repository),
    )

    final_state = asyncio.run(
        graph.run(SourceLearningState(source_uuid='source-1'))
    )

    assert 'old-card' not in repository.cards
    assert final_state.atomic_flashcard_count == (2 if has_evidence else 0)
    assert final_state.coherent_flashcard_count == (1 if has_evidence else 0)
    if not has_evidence:
        assert repository.cards == {}
        return

    coherent = final_state.coherent_occurrences[0]
    atomic = [
        occurrence
        for packet in final_state.atomic_packets
        for occurrence in packet.cards
    ]
    assert set(repository.cards) == {
        coherent.card.uuid,
        *(occurrence.card.uuid for occurrence in atomic),
    }
    assert [occurrence.source_block_uuids for occurrence in atomic] == [
        ['block-1'],
        ['block-2'],
    ]
    assert coherent.derived_card_uuids == [
        occurrence.card.uuid for occurrence in atomic
    ]
    assert coherent.source_block_uuids == [
        'block-1',
        'block-2',
    ]


def test_coherent_model_input_rebuilds_shared_catalog_for_atomic_packets(
    tmp_path,
):
    asset_path = tmp_path / 'shared.png'
    Image.new('RGB', (8, 8), (0, 0, 255)).save(asset_path)
    packets = []
    for index in range(2):
        block = SourceBlock(
            uuid=f'visual-owner-{index}',
            block_type=BlockType.IMAGE,
            content=None,
            assets=[
                VisualAsset(
                    uuid=f'visual-asset-{index}',
                    path=str(asset_path),
                )
            ],
        )
        request = SourceAtomicFlashcardRequest(
            source_block_uuid=block.uuid,
            source_page_index=0,
            window=SourceBlockWindow(target=[block]),
        )
        packets.append(
            SourceAtomicFlashcardPacket(
                request=request,
                cards=[
                    SourceFlashcardOccurrence(
                        card=SourceFlashcard(
                            question=f'Question {index}',
                            answer=f'Answer {index}',
                        ),
                        source_block_uuids=[block.uuid],
                    )
                ],
            )
        )
    request = SourceCoherentFlashcardRequest(
        source_page_index=0,
        packets=packets,
    )
    builder = MultimodalContentBuilder()

    model_input = request.model_input(builder)

    assert model_input.packets[0].evidence.source_text is None
    assert model_input.packets[1].evidence.source_text is None
    assert model_input.packets[0].evidence.source_context.target_blocks[
        0
    ].material.image_positions == [0]
    assert model_input.packets[1].evidence.source_context.target_blocks[
        0
    ].material.image_positions == [0]
    assert len(builder.images) == 1


def test_atomic_image_only_owner_projects_and_loads_native_source_image(
    tmp_path,
):
    asset_path = tmp_path / 'owner.png'
    Image.new('RGB', (8, 8), (255, 0, 0)).save(asset_path)
    owner = SourceBlock(
        uuid='visual-owner',
        block_type=BlockType.IMAGE,
        content=None,
        assets=[VisualAsset(uuid='visual-asset', path=str(asset_path))],
    )
    request = SourceAtomicFlashcardRequest(
        source_block_uuid=owner.uuid,
        source_page_index=0,
        window=SourceBlockWindow(target=[owner]),
    )

    class ImageModule:
        async def acall(self, *, request, content_images):
            self.request = request
            self.content_images = content_images
            return [
                SourceFlashcardCandidate(
                    question='What is shown?',
                    answer='The referenced visual.',
                )
            ]

    module = ImageModule()
    node = SourceAtomicFlashcardNode(
        _Repository([request]),
        module,
        ContextWindowSettings(backward_budget=0, forward_budget=0),
        _BlockRepository([owner]),
        budget=TokenBudget(
            counter=MaterialTokenEstimator(_Counter()), token_limit=100
        ),
        image_budget=ImagePayloadBudget(1, 26214400, 40000000),
    )

    asyncio.run(node.worker({'ordinal': 0, 'request': request}))

    evidence = module.request.evidence
    assert evidence.source_text is None
    assert evidence.source_context.target_blocks[
        0
    ].material.image_positions == [0]
    assert len(module.content_images) == 1


def _save_image(path, color):
    Image.new('RGB', (8, 8), color).save(path)
    return path


def _pixel(payload):
    with Image.open(BytesIO(payload)) as image:
        return image.getpixel((0, 0))


def _visual_packet(index, path, *, card_count=2):
    block = SourceBlock(
        uuid=f'visual-owner-{index}',
        block_type=BlockType.IMAGE,
        content=None,
        assets=[
            VisualAsset(
                uuid=f'visual-asset-{index}',
                path=str(path),
            )
        ],
    )
    request = SourceAtomicFlashcardRequest(
        source_block_uuid=block.uuid,
        source_page_index=0,
        window=SourceBlockWindow(target=[block]),
    )
    return SourceAtomicFlashcardPacket(
        request=request,
        cards=[
            SourceFlashcardOccurrence(
                card=SourceFlashcard(
                    uuid=f'card-{index}-{card_index}',
                    question=f'Question {index}-{card_index}',
                    answer=f'Answer {index}-{card_index}',
                ),
                source_block_uuids=[block.uuid],
            )
            for card_index in range(card_count)
        ],
    )


def test_atomic_worker_maps_neighbor_and_target_images_in_window_order(
    tmp_path, monkeypatch
):
    red_path = _save_image(tmp_path / 'context-red.png', (255, 0, 0))
    blue_path = _save_image(tmp_path / 'target-blue.png', (0, 0, 255))
    context = SourceBlock(
        uuid='red-context',
        block_type=BlockType.IMAGE,
        content=None,
        assets=[VisualAsset(uuid='red-asset', path=str(red_path))],
    )
    target = SourceBlock(
        uuid='text-owner',
        block_type=BlockType.PARAGRAPH,
        content='The text owner describes its nearby visual.',
        assets=[VisualAsset(uuid='blue-asset', path=str(blue_path))],
    )
    backend_request = SourceAtomicFlashcardRequest(
        source_block_uuid=target.uuid,
        source_page_index=0,
        window=SourceBlockWindow(target=[target]),
    )
    module = _AtomicModule()
    node = SourceAtomicFlashcardNode(
        _Repository([backend_request]),
        module,
        ContextWindowSettings(backward_budget=2, forward_budget=0),
        _BlockRepository([context, target]),
        budget=TokenBudget(
            counter=MaterialTokenEstimator(_Counter()), token_limit=100
        ),
        image_budget=ImagePayloadBudget(2, 26214400, 40000000),
    )
    monkeypatch.setattr(atomic_node.dspy, 'Image', lambda payload: payload)

    loaded = asyncio.run(
        node.load(SourceLearningState(source_uuid='source-1'))
    )['atomic_requests'][0]
    assert loaded.window.context_before == [context]
    asyncio.run(node.worker({'ordinal': 0, 'request': loaded}))

    evidence, images = module.calls[0]
    assert evidence.evidence.source_text == target.content
    assert evidence.evidence.source_context.context_before[
        0
    ].material.image_positions == [0]
    assert evidence.evidence.source_context.target_blocks[
        0
    ].material.image_positions == [1]
    assert [_pixel(payload) for payload in images] == [
        (255, 0, 0),
        (0, 0, 255),
    ]


def test_coherent_worker_rebuilds_shared_catalog_and_preserves_provenance(
    tmp_path, monkeypatch
):
    red_path = _save_image(tmp_path / 'first.png', (255, 0, 0))
    blue_path = _save_image(tmp_path / 'second.png', (0, 0, 255))
    packets = [
        _visual_packet(1, red_path),
        _visual_packet(2, blue_path),
    ]
    state = SourceLearningState(source_uuid='source-1', atomic_packets=packets)
    module = _CoherentModule(card_indexes=(1, 4))
    node = SourceCoherentFlashcardNode(
        module,
        budget=TokenBudget(
            counter=MaterialTokenEstimator(_Counter()), token_limit=100
        ),
        image_budget=ImagePayloadBudget(2, 26214400, 40000000),
    )
    monkeypatch.setattr(atomic_node.dspy, 'Image', lambda payload: payload)

    prepared = asyncio.run(node.prepare(state))
    request = prepared['coherent_requests'][0]
    result = asyncio.run(node.worker({'ordinal': 0, 'request': request}))
    occurrences = node.collect(
        state.model_copy(
            update={
                **prepared,
                'coherent_results': result['coherent_results'],
            }
        )
    )['coherent_occurrences']

    model_input = module.inputs[0]
    assert [card.index for card in model_input.packets[0].cards] == [1, 2]
    assert [card.index for card in model_input.packets[1].cards] == [3, 4]
    assert [
        packet.evidence.source_context.target_blocks[0].material.image_positions
        for packet in model_input.packets
    ] == [[0], [1]]
    assert [_pixel(payload) for payload in module.images[0]] == [
        (255, 0, 0),
        (0, 0, 255),
    ]
    assert len(occurrences) == 1
    assert occurrences[0].source_block_uuids == [
        'visual-owner-1',
        'visual-owner-2',
    ]
    assert occurrences[0].derived_card_uuids == [
        packets[0].cards[0].card.uuid,
        packets[1].cards[1].card.uuid,
    ]


def test_coherent_image_budget_splits_whole_packets_without_index_drift(
    tmp_path, monkeypatch
):
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
    ]
    packets = [
        _visual_packet(
            index,
            _save_image(tmp_path / f'packet-{index}.png', color),
        )
        for index, color in enumerate(colors)
    ]
    state = SourceLearningState(source_uuid='source-1', atomic_packets=packets)
    module = _CoherentModule(card_indexes=(1, 4))
    node = SourceCoherentFlashcardNode(
        module,
        budget=TokenBudget(
            counter=MaterialTokenEstimator(_Counter()), token_limit=100
        ),
        image_budget=ImagePayloadBudget(2, 26214400, 40000000),
    )
    monkeypatch.setattr(atomic_node.dspy, 'Image', lambda payload: payload)

    prepared = asyncio.run(node.prepare(state))
    requests = prepared['coherent_requests']
    results = [
        asyncio.run(node.worker({'ordinal': ordinal, 'request': request}))
        for ordinal, request in enumerate(requests)
    ]
    occurrences = node.collect(
        state.model_copy(
            update={
                **prepared,
                'coherent_results': [
                    result['coherent_results'][0] for result in results
                ],
            }
        )
    )['coherent_occurrences']

    assert [len(request.packets) for request in requests] == [2, 2]
    assert [
        [card.index for packet in call.packets for card in packet.cards]
        for call in module.inputs
    ] == [[1, 2, 3, 4], [1, 2, 3, 4]]
    assert [
        [
            positions
            for packet in call.packets
            for positions in [
                block.material.image_positions
                for block in packet.evidence.source_context.target_blocks
            ]
        ]
        for call in module.inputs
    ] == [[[0], [1]], [[0], [1]]]
    assert [
        [_pixel(payload) for payload in images] for images in module.images
    ] == [colors[:2], colors[2:]]
    assert [occurrence.source_block_uuids for occurrence in occurrences] == [
        ['visual-owner-0', 'visual-owner-1'],
        ['visual-owner-2', 'visual-owner-3'],
    ]
    assert [occurrence.derived_card_uuids for occurrence in occurrences] == [
        [packets[0].cards[0].card.uuid, packets[1].cards[1].card.uuid],
        [packets[2].cards[0].card.uuid, packets[3].cards[1].card.uuid],
    ]


def test_atomic_singleton_visual_context_over_budget_precedes_prediction(
    tmp_path,
):
    image_path = _save_image(tmp_path / 'required-context.png', (0, 255, 0))
    context = SourceBlock(
        uuid='required-context',
        block_type=BlockType.IMAGE,
        content=None,
        assets=[VisualAsset(uuid='required-asset', path=str(image_path))],
    )
    target = SourceBlock(
        uuid='text-target',
        block_type=BlockType.PARAGRAPH,
        content='A target referring to the required visual.',
    )
    request = SourceAtomicFlashcardRequest(
        source_block_uuid=target.uuid,
        source_page_index=0,
        window=SourceBlockWindow(
            context_before=[context],
            target=[target],
        ),
    )
    module = _AtomicModule()
    node = SourceAtomicFlashcardNode(
        _Repository([request]),
        module,
        ContextWindowSettings(backward_budget=0, forward_budget=0),
        _BlockRepository([context, target]),
        budget=TokenBudget(
            counter=MaterialTokenEstimator(_Counter()), token_limit=100
        ),
        image_budget=ImagePayloadBudget(0, 26214400, 40000000),
    )

    with pytest.raises(InputBudgetExceeded):
        asyncio.run(
            node.dispatch(
                SourceLearningState(
                    source_uuid='source-1',
                    atomic_requests=[request],
                )
            )
        )

    assert module.calls == []


def test_ineligible_coherent_single_card_does_not_project_or_read_images(
    tmp_path, monkeypatch
):
    packet = _visual_packet(
        1,
        _save_image(tmp_path / 'discarded.png', (255, 0, 0)),
        card_count=1,
    )
    module = _CoherentModule()
    node = SourceCoherentFlashcardNode(
        module,
        budget=TokenBudget(
            counter=MaterialTokenEstimator(_Counter()), token_limit=100
        ),
        image_budget=ImagePayloadBudget(0, 26214400, 40000000),
    )

    def fail_projection(self, **kwargs):
        raise AssertionError('ineligible packet must not be projected')

    def fail_image_inspection(*args, **kwargs):
        raise AssertionError('ineligible packet must not inspect image files')

    monkeypatch.setattr(
        content_service.MultimodalContentBuilder, 'build', fail_projection
    )
    monkeypatch.setattr(
        content_service, '_image_metadata_fits', fail_image_inspection
    )
    requests = asyncio.run(
        node.prepare(
            SourceLearningState(source_uuid='source-1', atomic_packets=[packet])
        )
    )['coherent_requests']

    assert requests == []
    assert module.inputs == []


def test_eligible_indivisible_coherent_packet_over_budget_precedes_prediction(
    tmp_path,
):
    packet = _visual_packet(
        1,
        _save_image(tmp_path / 'required.png', (0, 0, 255)),
        card_count=2,
    )
    module = _CoherentModule()
    node = SourceCoherentFlashcardNode(
        module,
        budget=TokenBudget(
            counter=MaterialTokenEstimator(_Counter()), token_limit=100
        ),
        image_budget=ImagePayloadBudget(0, 26214400, 40000000),
    )

    with pytest.raises(InputBudgetExceeded):
        asyncio.run(
            node.prepare(
                SourceLearningState(
                    source_uuid='source-1',
                    atomic_packets=[packet],
                )
            )
        )

    assert module.inputs == []
    assert module.images == []


def test_repository_returns_image_only_owner_as_metadata():
    row = {
        'source_block_uuid': 'image-only-owner',
        'source_page_index': 3,
        'owner_position': 7,
    }

    class Result:
        async def data(self):
            return [row]

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def run(self, query, **parameters):
            self.query = query
            self.parameters = parameters
            return Result()

    session = Session()
    repository = SourceLearningRepository(lambda: session)

    materials = asyncio.run(
        repository.load_atomic_flashcard_materials('source-1')
    )

    assert [material.model_dump() for material in materials] == [
        {
            'source_block_uuid': 'image-only-owner',
            'source_page_index': 3,
        }
    ]
    assert 'EXISTS {' in session.query
    assert session.parameters == {'source_uuid': 'source-1'}
