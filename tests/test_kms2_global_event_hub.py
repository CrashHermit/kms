import asyncio

from langgraph.graph import END, START, StateGraph

from kms2.config.global_semantic import GlobalEventHubSettings
from kms2.core.model.global_semantic.global_event_hub import (
    GlobalEventHub,
    GlobalEventHubCandidate,
    GlobalEventHubDefinition,
    GlobalEventHubJudgeDecision,
    GlobalEventHubJudgeInput,
    GlobalEventHubMember,
)
from kms2.langgraph.global_semantic.global_event_hub import (
    add_global_event_hub_phase,
)
from kms2.langgraph.global_semantic.state import GlobalSemanticState
from kms2.node.global_semantic.global_event_hub import GlobalEventHubNode
from kms2.node.global_semantic.global_event_hub_persistence import (
    GlobalEventHubPersistenceNode,
)


def _candidate():
    return GlobalEventHubCandidate(
        left_uuid='hub-1',
        left_name='Launch',
        left_description='begins rollout',
        right_uuid='hub-2',
        right_name='Launch',
        right_description='begins rollout',
        score=0.9,
    )


def test_global_event_hub_model_uses_automatic_uuid_and_no_membership_property():
    hub = GlobalEventHub(
        name='Launch',
        aliases=['Launch'],
        description='begins rollout',
        embedding=[0.7],
    )

    assert hub.uuid
    assert 'source_uuid' not in GlobalEventHub.model_fields
    assert 'membership_uuids' not in GlobalEventHub.model_fields


class _Repository:
    def __init__(self):
        self.accepted = None

    async def read_global_event_hub_candidates(self, **kwargs):
        return [_candidate()]

    async def replace_global_event_hub_accepted_edges(self, pairs):
        self.accepted = pairs

    async def detect_global_event_hub_communities(self, **kwargs):
        return [
            [
                GlobalEventHubMember(
                    uuid='hub-1',
                    name='Launch',
                    description='begins rollout',
                ),
                GlobalEventHubMember(
                    uuid='hub-2',
                    name='Launch',
                    description='begins rollout',
                ),
            ]
        ]


class _Synthesis:
    def __init__(self):
        self.requests = []

    async def aforward(self, *, request):
        self.requests.append(request)
        return GlobalEventHubDefinition(
            name='Launch', description='begins rollout'
        )


class _Embedding:
    async def embed(self, texts):
        assert list(texts) == ['Launch: begins rollout']
        return [[0.7]]


class _Reranker:
    def __init__(self):
        self.calls = []

    async def rerank(self, query, documents, top_n=None):
        self.calls.append((query, documents, top_n))
        return [{'index': 0, 'relevance_score': 0.6}]


class _Judge:
    def __init__(self):
        self.requests = []

    async def aforward(self, *, requests: list[GlobalEventHubJudgeInput]):
        self.requests.append(requests)
        return [
            GlobalEventHubJudgeDecision(index=index, belongs_in_same_hub=True)
            for index, _ in enumerate(requests)
        ]


def test_global_event_node_uses_canonical_event_evidence_and_outputs_membership():
    repository = _Repository()
    synthesis = _Synthesis()
    reranker = _Reranker()
    judge = _Judge()
    node = GlobalEventHubNode(
        repository,
        synthesis,
        judge,
        reranker,
        _Embedding(),
        GlobalEventHubSettings(),
    )

    state = GlobalSemanticState(
        **asyncio.run(node.load_candidates(GlobalSemanticState()))
    )
    sends = node.dispatch_rerank(state)
    rerank_result = asyncio.run(node.rerank_worker(sends[0].arg))
    state = state.model_copy(update=rerank_result)
    state = state.model_copy(update=node.collect_rerank(state))
    judge_sends = node.dispatch_judge(state)
    judge_result = asyncio.run(node.judge_worker(judge_sends[0].arg))
    state = state.model_copy(update=judge_result)
    state = state.model_copy(update=node.collect_judge(state))
    state = state.model_copy(update=asyncio.run(node.detect_communities(state)))
    synthesis_sends = node.dispatch_synthesis(state)
    result = asyncio.run(node.synthesis_worker(synthesis_sends[0].arg))
    state = state.model_copy(update=result)
    state = state.model_copy(update=node.collect_synthesis(state))
    embedded = asyncio.run(node.embed(state))

    assert reranker.calls == [
        ('Launch: begins rollout', ['Launch: begins rollout'], None)
    ]
    request = judge.requests[0][0]
    assert request.left_name == 'Launch'
    assert request.left_description == 'begins rollout'
    assert request.right_name == 'Launch'
    assert request.right_description == 'begins rollout'
    assert synthesis.requests[0].members[0].name == 'Launch'
    assert synthesis.requests[0].members[0].description == 'begins rollout'
    assert repository.accepted == [_candidate()]
    assert embedded['global_event_hubs'][0].name == 'Launch'
    assert embedded['global_event_hubs'][0].aliases == ['Launch', 'Launch']
    assert embedded['global_event_hubs'][0].embedding == [0.7]
    assert embedded['global_event_hub_memberships'] == [['hub-1', 'hub-2']]


def test_global_event_phase_exposes_complete_topology():
    node = GlobalEventHubNode(
        _Repository(),
        _Synthesis(),
        _Judge(),
        _Reranker(),
        _Embedding(),
        GlobalEventHubSettings(),
    )
    graph = StateGraph(GlobalSemanticState)
    add_global_event_hub_phase(
        graph,
        node,
        GlobalEventHubPersistenceNode(_Repository()),
    )
    graph.add_edge(START, 'global_event_hub_load')
    graph.add_edge('global_event_hub_persistence', END)
    compiled = graph.compile()

    assert {
        'global_event_hub_load',
        'global_event_hub_rerank_worker',
        'global_event_hub_rerank_collect',
        'global_event_hub_judge_worker',
        'global_event_hub_judge_collect',
        'global_event_hub_communities',
        'global_event_hub_synthesis_worker',
        'global_event_hub_synthesis_collect',
        'global_event_hub_embedding',
        'global_event_hub_persistence',
    } <= set(compiled.nodes)
