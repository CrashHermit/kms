import asyncio
import base64
from io import BytesIO

import dspy
import pytest
from kms_token_helpers import JUDGE_BUDGET
from langgraph.graph import StateGraph
from PIL import Image

from kms.config.inference import ContextWindowSettings
from kms.config.source_semantic import SourceEntityHubSettings
from kms.core.media_budget import ImagePayloadBudget
from kms.core.model.block import SourceBlock
from kms.core.model.block_types import BlockType
from kms.core.model.context import SourceBlockWindow
from kms.core.model.source_semantic.source_entity_hub import (
    SourceEntityHubCandidate,
    SourceEntityHubJudgeDecision,
    SourceEntityHubJudgeInput,
    SourceEntityHubMember,
)
from kms.core.model.visual_asset import VisualAsset
from kms.core.tokenization.budget import InputBudgetExceeded, TokenBudget
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.langgraph.source_semantic.entity_hub.graph import SourceEntityHubGraph
from kms.langgraph.source_semantic.source_entity_hub import (
    add_source_entity_hub_phase,
)
from kms.langgraph.source_semantic.state import SourceSemanticState
from kms.node.source_semantic.source_entity_hub import SourceEntityHubNode
from kms.node.source_semantic.source_entity_hub_persistence import (
    SourceEntityHubPersistenceNode,
)
from kms.node.source_semantic.source_entity_hub_reconciliation import (
    SourceEntityHubReconciliationNode,
)


def _window(uuid: str, target: str, context: str = '') -> SourceBlockWindow:
    return SourceBlockWindow(
        context_before=(
            [
                SourceBlock(
                    uuid=f'context-{uuid}',
                    block_type='paragraph',
                    content=context,
                )
            ]
            if context
            else []
        ),
        target=[
            SourceBlock(
                uuid=f'block-{uuid}',
                block_type='paragraph',
                content=target,
            )
        ],
    )


class _LengthCounter:
    def count_texts(self, texts):
        return [len(text) for text in texts]


def _member(
    uuid: str, *, definition: str | None = None
) -> SourceEntityHubMember:
    text = definition or f'Use f in exercise {uuid}.'
    return SourceEntityHubMember(
        uuid=uuid,
        source_block_uuid=f'block-{uuid}',
        name='f',
        description=definition or 'the function',
        window=_window(uuid, text, f'Exercise {uuid}.'),
    )


def _candidate(
    left: SourceEntityHubMember,
    right: SourceEntityHubMember,
    left_group: str | None = None,
    right_group: str | None = None,
) -> SourceEntityHubCandidate:
    return SourceEntityHubCandidate(
        left_group_uuid=left_group or left.uuid,
        right_group_uuid=right_group or right.uuid,
        left_uuid=left.uuid,
        left_name=left.name,
        left_description=left.description,
        left_window=left.window,
        right_uuid=right.uuid,
        right_name=right.name,
        right_description=right.description,
        right_window=right.window,
        score=0.99,
    )


def _source_blocks(members):
    return [member.window.target[0] for member in members]


def _state(**updates):
    return SourceSemanticState(source_uuid='source-1', **updates)


class _Repository:
    def __init__(self, members, proposals_by_group_count=None):
        self.members = members
        self.proposals_by_group_count = proposals_by_group_count or {}
        self.group_snapshots = []

    async def read_source_entity_hub_members(self, source_uuid):
        return self.members

    async def read_source_entity_hub_candidates(
        self, source_uuid, groups, *, candidate_limit, minimum_similarity
    ):
        self.group_snapshots.append(groups)
        return self.proposals_by_group_count.get(len(groups), [])

    async def replace_source_entity_hubs(self, source_uuid, hubs, memberships):
        self.memberships = memberships


class _Reranker:
    def __init__(self, score=1.0):
        self.score = score
        self.calls = []

    async def rerank(self, query, documents, *, content_images, top_n=None):
        assert query.text is not None
        assert all(document.text is not None for document in documents)
        assert content_images == []
        self.calls.append(
            (query.text, [document.text for document in documents], top_n)
        )
        return [
            {'index': index, 'relevance_score': self.score}
            for index, _ in enumerate(documents)
        ]


class _Judge:
    def __init__(self, answers=None):
        self.answers = answers or []
        self.requests = []

    async def acall(
        self,
        *,
        requests: list[SourceEntityHubJudgeInput],
        content_images,
    ):
        assert content_images == []
        offset = len(self.requests)
        self.requests.extend(requests)
        decisions = []
        for index, _ in enumerate(requests):
            answer_index = offset + index
            belongs = (
                self.answers[answer_index]
                if answer_index < len(self.answers)
                else True
            )
            decisions.append(
                SourceEntityHubJudgeDecision(
                    index=index,
                    belongs_in_same_hub=belongs,
                )
            )
        return decisions


def _node(repository, judge, reranker=None, settings=None):
    return SourceEntityHubNode(
        repository,
        judge,
        reranker or _Reranker(),
        settings or SourceEntityHubSettings(),
        judge_budget=JUDGE_BUDGET,
        image_budget=ImagePayloadBudget(4, 26_214_400, 40_000_000),
    )


async def _round(node, state):
    if (
        not state.source_entity_hub_members
        and not state.source_entity_hub_groups
    ):
        state = state.model_copy(update=await node.initialize(state))
    state = state.model_copy(update=await node.load_candidates(state))
    dispatched = node.dispatch_rerank(state)
    rerank_results = []
    if not isinstance(dispatched, str):
        for send in dispatched:
            rerank_results.extend(
                (await node.rerank_worker(send.arg))[
                    'source_entity_hub_rerank_results'
                ]
            )
    state = state.model_copy(
        update={'source_entity_hub_rerank_results': rerank_results}
    )
    rerank_update = node.collect_rerank(state)
    rerank_update['source_entity_hub_rerank_results'] = []
    state = state.model_copy(update=rerank_update)
    dispatched = await node.dispatch_judge(state)
    judge_results = []
    if not isinstance(dispatched, str):
        for send in dispatched:
            judge_results.extend(
                (await node.judge_worker(send.arg))[
                    'source_entity_hub_judge_results'
                ]
            )
    state = state.model_copy(
        update={'source_entity_hub_judge_results': judge_results}
    )
    judge_update = node.collect_judge(state)
    judge_update['source_entity_hub_judge_results'] = []
    state = state.model_copy(update=judge_update)
    return state.model_copy(update=node.merge_groups(state))


def test_merging_reopens_group_candidate_discovery_and_reuses_pair_decisions():
    a, b, c = (_member(name) for name in ('a', 'b', 'c'))
    repository = _Repository(
        [a, b, c],
        {
            3: [_candidate(a, b)],
            2: [_candidate(a, c, 'a', 'c')],
        },
    )
    judge = _Judge()
    node = _node(repository, judge)

    async def run():
        graph = StateGraph(SourceSemanticState)
        add_source_entity_hub_phase(
            graph,
            SourceEntityHubReconciliationNode(node, SourceEntityHubGraph(node)),
            SourceEntityHubPersistenceNode(repository),
        )
        graph.set_entry_point('source_entity_hub_load')
        graph.set_finish_point('source_entity_hub_persistence')
        return await graph.compile().ainvoke(
            _state(blocks=_source_blocks([a, b, c]))
        )

    result = asyncio.run(run())

    assert repository.group_snapshots == [
        [[a], [b], [c]],
        [[a, b], [c]],
        [[a, b, c]],
    ]
    assert result['source_entity_hub_memberships'] == [['a', 'b', 'c']]
    assert repository.memberships == [['a', 'b', 'c']]
    assert len(judge.requests) == 3
    assert result['source_entity_hub_count'] == 1


def test_high_reranker_score_never_bypasses_judge():
    a, b = _member('a'), _member('b')
    repository = _Repository([a, b], {2: [_candidate(a, b)]})
    judge = _Judge([False])
    node = _node(repository, judge, _Reranker(score=1.0))

    state = asyncio.run(_round(node, _state(blocks=_source_blocks([a, b]))))

    assert len(judge.requests) == 1
    assert state.source_entity_hub_groups == [[a], [b]]
    assert state.source_entity_hub_decision_cache['a:b'] is False


def test_unknown_cross_member_pair_does_not_authorize_a_group_merge():
    a, b, c = _member('a'), _member('b'), _member('c')
    node = _node(_Repository([a, b, c]), _Judge())
    state = _state(
        source_entity_hub_members=[a, b, c],
        source_entity_hub_groups=[[a, b], [c]],
        source_entity_hub_candidates=[
            _candidate(a, c, 'a', 'c'),
            _candidate(b, c, 'a', 'c'),
        ],
        source_entity_hub_decision_cache={'a:c': True},
    )

    result = node.merge_groups(state)

    assert result['source_entity_hub_groups'] == [[a, b], [c]]
    assert result['source_entity_hub_round_merged'] is False
    assert state.source_entity_hub_decision_cache == {'a:c': True}


def test_negative_cross_member_judgment_prevents_group_merge():
    a, b, c = _member('a'), _member('b'), _member('c')
    repository = _Repository([a, b, c], {2: [_candidate(a, b, 'a', 'c')]})
    judge = _Judge([True, False])
    node = _node(repository, judge)
    state = _state(
        source_entity_hub_members=[a, b, c],
        source_entity_hub_groups=[[a, b], [c]],
    )

    state = asyncio.run(_round(node, state))

    assert state.source_entity_hub_groups == [[a, b], [c]]
    assert state.source_entity_hub_decision_cache['a:c'] is True
    assert state.source_entity_hub_decision_cache['b:c'] is False


def test_cached_pair_skips_repeated_reranker_and_judge_work():
    a, b = _member('a'), _member('b')
    repository = _Repository([a, b], {2: [_candidate(a, b)]})
    judge = _Judge()
    reranker = _Reranker()
    node = _node(repository, judge, reranker)
    state = _state(
        source_entity_hub_members=[a, b],
        source_entity_hub_groups=[[a], [b]],
        source_entity_hub_decision_cache={'a:b': True},
    )

    state = state.model_copy(update=asyncio.run(node.load_candidates(state)))

    assert node.dispatch_rerank(state) == 'source_entity_hub_rerank_collect'
    assert reranker.calls == []
    assert judge.requests == []


def test_identity_judge_receives_uuid_free_consumer_selected_evidence():
    a = _member('exercise-1', definition='f(x) = x squared')
    b = _member('exercise-2', definition='f denotes force')
    repository = _Repository([a, b], {2: [_candidate(a, b)]})
    judge = _Judge([False])
    node = _node(repository, judge)
    blocks = [
        SourceBlock(
            uuid='block-exercise-1',
            block_type='paragraph',
            content='f(x) = x squared',
        ),
        SourceBlock(
            uuid='block-exercise-2',
            block_type='paragraph',
            content='f denotes force',
        ),
    ]

    asyncio.run(_round(node, _state(blocks=blocks)))

    request = judge.requests[0]
    assert (
        request.left_evidence.target_blocks[0].material.text
        == 'f(x) = x squared'
    )
    assert (
        request.right_evidence.target_blocks[0].material.text
        == 'f denotes force'
    )
    assert 'left_uuid' not in request.model_dump()
    assert 'right_uuid' not in request.model_dump()


def test_hub_judge_uses_ordered_context_with_its_own_token_budget():
    member = _member('middle', definition='target')
    other = _member('far-after', definition='dddd')
    settings = SourceEntityHubSettings(
        context_window=ContextWindowSettings(
            backward_budget=3, forward_budget=2
        )
    )

    class Repository(_Repository):
        async def read_source_entity_hub_candidates(
            self, source_uuid, groups, *, candidate_limit, minimum_similarity
        ):
            return [_candidate(groups[0][0], groups[1][0])]

    repository = Repository([member, other])
    judge = _Judge()
    node = _node(repository, judge, settings=settings)
    node._judge_budget = TokenBudget(
        MaterialTokenEstimator(_LengthCounter()), 10_000
    )
    blocks = [
        SourceBlock(uuid='far-before', block_type='paragraph', content='aa'),
        SourceBlock(uuid='near-before', block_type='paragraph', content='bbb'),
        SourceBlock(
            uuid='block-middle', block_type='paragraph', content='target'
        ),
        SourceBlock(uuid='near-after', block_type='paragraph', content='cc'),
        SourceBlock(
            uuid='block-far-after', block_type='paragraph', content='dddd'
        ),
    ]

    asyncio.run(_round(node, _state(blocks=blocks)))

    request = judge.requests[0]
    assert [
        block.material.text for block in request.left_evidence.context_before
    ] == ['bbb']
    assert request.left_evidence.target_blocks[0].material.text == 'target'
    assert [
        block.material.text for block in request.left_evidence.context_after
    ] == ['cc']
    assert [
        block.material.text for block in request.right_evidence.context_before
    ] == ['cc']
    assert request.right_evidence.target_blocks[0].material.text == 'dddd'
    assert 'left_uuid' not in request.model_dump()
    assert 'right_uuid' not in request.model_dump()


def test_empty_and_singleton_inputs_produce_deterministic_disjoint_hubs():
    empty_node = _node(_Repository([]), _Judge())
    empty = asyncio.run(_round(empty_node, _state()))
    assert empty.source_entity_hub_groups == []
    assert empty_node.repeat_round(empty) == 'source_entity_hub_prepare'
    assert empty_node.prepare_hubs(empty)['source_entity_hub_memberships'] == []

    members = [_member('b'), _member('a')]
    singleton_node = _node(_Repository(members), _Judge())
    singleton = asyncio.run(
        _round(singleton_node, _state(blocks=_source_blocks(members)))
    )
    hubs = singleton_node.prepare_hubs(singleton)
    assert hubs['source_entity_hub_memberships'] == [['a'], ['b']]
    assert (
        len(
            {
                item
                for group in hubs['source_entity_hub_memberships']
                for item in group
            }
        )
        == 2
    )


class _FixedCounter:
    def count_texts(self, texts):
        return [4 * text.count('"left_name"') for text in texts]


@pytest.mark.parametrize(
    ('batch_size', 'expected_sizes'),
    [(1, [1, 1, 1]), (10, [2, 1])],
)
def test_entity_judge_respects_token_and_item_limits(
    batch_size, expected_sizes
):
    a, b, c, d = (_member(name) for name in 'abcd')
    node = _node(
        _Repository([a, b, c, d]),
        _Judge(),
        settings=SourceEntityHubSettings(judge_batch_size=batch_size),
    )
    node._judge_budget = TokenBudget(MaterialTokenEstimator(_FixedCounter()), 8)
    state = _state(
        source_entity_hub_reranked_candidates=[
            _candidate(a, b),
            _candidate(a, c),
            _candidate(a, d),
        ]
    )

    sends = asyncio.run(node.dispatch_judge(state))

    assert [
        len(send.arg['source_entity_hub_batch']) for send in sends
    ] == expected_sizes
    assert [
        candidate.right_uuid
        for send in sends
        for candidate in send.arg['source_entity_hub_batch']
    ] == ['b', 'c', 'd']


def test_oversized_entity_judge_pair_is_not_dispatched():
    a, b = _member('a'), _member('b')
    judge = _Judge()
    node = _node(_Repository([a, b]), judge)
    node._judge_budget = TokenBudget(MaterialTokenEstimator(_FixedCounter()), 3)

    with pytest.raises(InputBudgetExceeded):
        asyncio.run(
            node.dispatch_judge(
                _state(source_entity_hub_reranked_candidates=[_candidate(a, b)])
            )
        )

    assert judge.requests == []


def test_entity_judge_keeps_left_right_images_distinct_when_batches_split(
    tmp_path,
):
    colors = {
        'red': (255, 0, 0),
        'blue': (0, 0, 255),
        'green': (0, 128, 0),
        'yellow': (255, 255, 0),
    }
    paths = {}
    for name, color in colors.items():
        path = tmp_path / f'{name}.png'
        Image.new('RGB', (8, 8), color).save(path)
        paths[name] = path

    members = {}
    for name in colors:
        members[name] = SourceEntityHubMember(
            uuid=name,
            source_block_uuid=f'block-{name}',
            name='f',
            description=name,
            window=SourceBlockWindow(
                target=[
                    SourceBlock(
                        uuid=f'block-{name}',
                        block_type=BlockType.IMAGE,
                        content=name,
                        assets=[
                            VisualAsset(
                                uuid=f'asset-{name}', path=str(paths[name])
                            )
                        ],
                    )
                ]
            ),
        )
    candidates = [
        _candidate(members['red'], members['blue']),
        _candidate(members['green'], members['yellow']),
    ]

    def image_color(image: dspy.Image) -> tuple[int, int, int]:
        payload = base64.b64decode(image.url.split(',', 1)[1])
        with Image.open(BytesIO(payload)) as opened:
            return opened.getpixel((0, 0))

    class ImageJudge:
        def __init__(self):
            self.calls = []

        async def acall(self, *, requests, content_images):
            request = requests[0]
            self.calls.append(
                (request, [image_color(image) for image in content_images])
            )
            assert len(requests) == 1
            assert request.index == 0
            assert request.left_evidence.target_blocks[
                0
            ].material.image_positions == [0]
            assert request.right_evidence.target_blocks[
                0
            ].material.image_positions == [1]
            return [
                SourceEntityHubJudgeDecision(
                    index=request.index, belongs_in_same_hub=False
                )
            ]

    judge = ImageJudge()
    node = _node(
        _Repository(list(members.values())),
        judge,
        settings=SourceEntityHubSettings(judge_batch_size=10),
    )
    node._judge_budget = TokenBudget(
        MaterialTokenEstimator(_LengthCounter()), 100_000
    )
    node._image_budget = ImagePayloadBudget(
        max_images=2, max_bytes=26_214_400, max_pixels=40_000_000
    )
    sends = asyncio.run(
        node.dispatch_judge(
            _state(source_entity_hub_reranked_candidates=candidates)
        )
    )
    assert [len(send.arg['source_entity_hub_batch']) for send in sends] == [
        1,
        1,
    ]
    for send in sends:
        asyncio.run(node.judge_worker(send.arg))
    assert [colors for _, colors in judge.calls] == [
        [colors['red'], colors['blue']],
        [colors['green'], colors['yellow']],
    ]
    assert [
        (
            request.left_description,
            request.right_description,
            request.left_evidence.target_blocks[0].material.text,
            request.right_evidence.target_blocks[0].material.text,
        )
        for request, _ in judge.calls
    ] == [
        ('red', 'blue', 'red', 'blue'),
        ('green', 'yellow', 'green', 'yellow'),
    ]
