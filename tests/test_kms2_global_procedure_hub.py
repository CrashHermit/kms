import asyncio

from kms2.config.global_semantic import GlobalProcedureHubSettings
from kms2.core.model.global_semantic.global_procedure_hub import (
    GlobalProcedureHub,
    GlobalProcedureHubCandidate,
    GlobalProcedureHubDefinition,
    GlobalProcedureHubJudgeDecision,
    GlobalProcedureHubJudgeInput,
    GlobalProcedureHubMember,
)
from kms2.langgraph.global_semantic.graph import GlobalSemanticGraph
from kms2.langgraph.global_semantic.state import GlobalSemanticState
from kms2.node.global_semantic.global_procedure_hub import (
    GlobalProcedureHubNode,
)
from kms2.node.global_semantic.global_procedure_hub_persistence import (
    GlobalProcedureHubPersistenceNode,
)


def _candidate():
    return GlobalProcedureHubCandidate(
        left_uuid='hub-1',
        left_description='authenticate a user',
        right_uuid='hub-2',
        right_description='authenticate a user',
        score=0.9,
    )


def test_global_procedure_hub_model_uses_automatic_uuid_and_no_membership_property():
    hub = GlobalProcedureHub(
        canonical_name='User authentication',
        aliases=['User authentication'],
        description='authenticate a user',
        embedding=[0.7],
    )

    assert hub.uuid
    assert 'source_uuid' not in GlobalProcedureHub.model_fields
    assert 'membership_uuids' not in GlobalProcedureHub.model_fields


class _Repository:
    def __init__(self):
        self.accepted = None

    async def read_global_procedure_hub_candidates(self, **kwargs):
        return [_candidate()]

    async def replace_global_procedure_hub_accepted_edges(self, pairs):
        self.accepted = pairs

    async def detect_global_procedure_hub_communities(self, **kwargs):
        return [
            [
                GlobalProcedureHubMember(
                    uuid='hub-1',
                    canonical_name='User authentication',
                    description='authenticate a user',
                ),
                GlobalProcedureHubMember(
                    uuid='hub-2',
                    canonical_name='Authenticate user',
                    description='authenticate a user',
                ),
            ]
        ]


class _Synthesis:
    def __init__(self):
        self.requests = []

    async def aforward(self, *, request):
        self.requests.append(request)
        return GlobalProcedureHubDefinition(
            canonical_name='User authentication',
            description='authenticate a user',
        )


class _Embedding:
    async def embed(self, texts):
        assert list(texts) == ['authenticate a user']
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

    async def aforward(self, *, requests: list[GlobalProcedureHubJudgeInput]):
        self.requests.append(requests)
        return [
            GlobalProcedureHubJudgeDecision(
                index=index, belongs_in_same_hub=True
            )
            for index, _ in enumerate(requests)
        ]


def test_global_procedure_node_uses_description_evidence_and_preserves_memberships():
    repository = _Repository()
    synthesis = _Synthesis()
    reranker = _Reranker()
    judge = _Judge()
    node = GlobalProcedureHubNode(
        repository,
        synthesis,
        judge,
        reranker,
        _Embedding(),
        GlobalProcedureHubSettings(),
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
        ('authenticate a user', ['authenticate a user'], None)
    ]
    request = judge.requests[0][0]
    assert request.left_description == 'authenticate a user'
    assert request.right_description == 'authenticate a user'
    assert not hasattr(request, 'left_canonical_name')
    assert repository.accepted == [_candidate()]
    assert synthesis.requests[0].members[0].description == 'authenticate a user'
    assert embedded['global_procedure_hubs'][0].uuid
    assert embedded['global_procedure_hubs'][0].aliases == [
        'User authentication',
        'Authenticate user',
    ]
    assert embedded['global_procedure_hub_memberships'] == [['hub-1', 'hub-2']]


class _TripletPhase:
    def load_groups(self, state):
        return {}

    def dispatch_synthesis(self, state):
        return 'global_triplet_hub_synthesis_collect'

    def synthesis_worker(self, state):
        return {}

    def collect_synthesis(self, state):
        return {}

    def embed(self, state):
        return {}

    async def run(self, state):
        return {}


def test_global_graph_exposes_complete_procedure_phase_topology():
    node = GlobalProcedureHubNode(
        _Repository(),
        _Synthesis(),
        _Judge(),
        _Reranker(),
        _Embedding(),
        GlobalProcedureHubSettings(),
    )
    triplet_phase = _TripletPhase()
    graph = GlobalSemanticGraph(
        node,
        GlobalProcedureHubPersistenceNode(_Repository()),
        node,
        GlobalProcedureHubPersistenceNode(_Repository()),
        node,
        GlobalProcedureHubPersistenceNode(_Repository()),
        triplet_phase,
        triplet_phase,
        triplet_phase,
        node,
        GlobalProcedureHubPersistenceNode(_Repository()),
        node,
        GlobalProcedureHubPersistenceNode(_Repository()),
    ).build_graph()

    assert {
        'global_procedure_hub_load',
        'global_procedure_hub_rerank_worker',
        'global_procedure_hub_rerank_collect',
        'global_procedure_hub_judge_worker',
        'global_procedure_hub_judge_collect',
        'global_procedure_hub_communities',
        'global_procedure_hub_synthesis_worker',
        'global_procedure_hub_synthesis_collect',
        'global_procedure_hub_embedding',
        'global_procedure_hub_persistence',
    } <= set(graph.nodes)
