import asyncio

from langgraph.graph import END, START, StateGraph

from kms2.config.global_semantic import GlobalEventHubSettings
from kms2.core.model import (
    GlobalEventHub,
    GlobalEventHubCandidate,
    GlobalEventHubDefinition,
    GlobalEventHubJudgeDecision,
    GlobalEventHubJudgeInput,
    GlobalEventHubMember,
)
from kms2.database.global_semantic.event_hub_repository import (
    GlobalEventHubRepository,
)
from kms2.database.global_semantic.queries.event_hub import (
    DETECT_GLOBAL_EVENT_HUB_COMMUNITIES,
    READ_GLOBAL_EVENT_HUB_CANDIDATES,
    REPLACE_GLOBAL_EVENT_HUB_ACCEPTED_EDGES,
)
from kms2.langgraph.global_semantic.event_hub import add_global_event_hub_phase
from kms2.langgraph.global_semantic.state import GlobalSemanticState
from kms2.node.global_semantic.event_hub import GlobalEventHubNode
from kms2.node.global_semantic.event_hub_persistence import (
    GlobalEventHubPersistenceNode,
)


class _Result:
    def __init__(self, rows=None):
        self.rows = rows or []

    async def data(self):
        return self.rows

    async def consume(self):
        return None


class _Context:
    def __init__(self, session):
        self.session = session

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, *args):
        return None


class _Session:
    def __init__(self):
        self.calls = []

    async def run(self, query, **parameters):
        self.calls.append((query, parameters))
        if query is READ_GLOBAL_EVENT_HUB_CANDIDATES:
            return _Result(
                [
                    {
                        'left_uuid': 'hub-1',
                        'left_name': 'Launch',
                        'left_description': 'begins rollout',
                        'right_uuid': 'hub-2',
                        'right_name': 'Launch',
                        'right_description': 'begins rollout',
                        'score': 0.9,
                    }
                ]
            )
        if query is DETECT_GLOBAL_EVENT_HUB_COMMUNITIES:
            return _Result(
                [
                    {
                        'community_id': 1,
                        'uuid': 'hub-1',
                        'name': 'Launch',
                        'description': 'begins rollout',
                    },
                    {
                        'community_id': 1,
                        'uuid': 'hub-2',
                        'name': 'Launch',
                        'description': 'begins rollout',
                    },
                ]
            )
        return _Result()


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


def test_global_event_repository_uses_event_only_cross_source_queries():
    session = _Session()
    repository = GlobalEventHubRepository(lambda: _Context(session))

    candidates = asyncio.run(
        repository.read_global_event_hub_candidates(
            candidate_limit=20,
            minimum_similarity=0.9,
        )
    )
    asyncio.run(repository.replace_global_event_hub_accepted_edges(candidates))
    communities = asyncio.run(
        repository.detect_global_event_hub_communities(
            max_iterations=10,
            min_association_strength=0.2,
            minimum_community_size=2,
        )
    )

    assert candidates == [_candidate()]
    assert len(communities) == 1
    query = session.calls[0][0]
    assert 'SourceEventHub' in query
    assert 'source_event_hub_embedding' in query
    assert 'candidate.source_uuid <> query.source_uuid' in query
    assert 'HAS_SUBJECT' not in query
    assert 'HAS_OBJECT' not in query
    assert 'source_uuid' not in session.calls[0][1]
    assert session.calls[1][0] is REPLACE_GLOBAL_EVENT_HUB_ACCEPTED_EDGES


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
