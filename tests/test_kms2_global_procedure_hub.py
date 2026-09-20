import asyncio

from kms2.config.global_semantic import GlobalProcedureHubSettings
from kms2.core.model import (
    GlobalProcedureHub,
    GlobalProcedureHubCandidate,
    GlobalProcedureHubDefinition,
    GlobalProcedureHubJudgeDecision,
    GlobalProcedureHubJudgeInput,
    GlobalProcedureHubMember,
)
from kms2.database.global_semantic.procedure_hub_repository import (
    GlobalProcedureHubRepository,
)
from kms2.database.global_semantic.queries.procedure_hub import (
    DETECT_GLOBAL_PROCEDURE_HUB_COMMUNITIES,
    READ_GLOBAL_PROCEDURE_HUB_CANDIDATES,
    REPLACE_GLOBAL_PROCEDURE_HUB_ACCEPTED_EDGES,
)
from kms2.langgraph.global_semantic.graph import GlobalSemanticGraph
from kms2.langgraph.global_semantic.state import GlobalSemanticState
from kms2.node.global_semantic.procedure_hub import GlobalProcedureHubNode
from kms2.node.global_semantic.procedure_hub_persistence import (
    GlobalProcedureHubPersistenceNode,
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
        if query is READ_GLOBAL_PROCEDURE_HUB_CANDIDATES:
            return _Result(
                [
                    {
                        'left_uuid': 'hub-1',
                        'left_description': 'authenticate a user',
                        'right_uuid': 'hub-2',
                        'right_description': 'authenticate a user',
                        'score': 0.9,
                    }
                ]
            )
        if query is DETECT_GLOBAL_PROCEDURE_HUB_COMMUNITIES:
            return _Result(
                [
                    {
                        'community_id': 1,
                        'uuid': 'hub-1',
                        'canonical_name': 'User authentication',
                        'description': 'authenticate a user',
                    },
                    {
                        'community_id': 1,
                        'uuid': 'hub-2',
                        'canonical_name': 'Authenticate user',
                        'description': 'authenticate a user',
                    },
                ]
            )
        return _Result()


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


def test_global_procedure_repository_uses_cross_source_procedure_queries():
    session = _Session()
    repository = GlobalProcedureHubRepository(lambda: _Context(session))

    candidates = asyncio.run(
        repository.read_global_procedure_hub_candidates(
            candidate_limit=20,
            minimum_similarity=0.9,
        )
    )
    asyncio.run(
        repository.replace_global_procedure_hub_accepted_edges(candidates)
    )
    communities = asyncio.run(
        repository.detect_global_procedure_hub_communities(
            max_iterations=10,
            min_association_strength=0.2,
            minimum_community_size=2,
        )
    )

    assert candidates == [_candidate()]
    assert len(communities) == 1
    query = session.calls[0][0]
    assert 'SourceProcedureHub' in query
    assert 'source_procedure_hub_embedding' in query
    assert 'candidate.source_uuid <> query.source_uuid' in query
    assert 'HAS_SUBJECT' not in query
    assert 'HAS_OBJECT' not in query
    assert 'SourceTripletHub' not in query
    assert 'source_uuid' not in session.calls[0][1]
    assert session.calls[1][0] is REPLACE_GLOBAL_PROCEDURE_HUB_ACCEPTED_EDGES
    assert session.calls[3][0] is DETECT_GLOBAL_PROCEDURE_HUB_COMMUNITIES


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


def test_global_graph_exposes_complete_procedure_phase_topology():
    node = GlobalProcedureHubNode(
        _Repository(),
        _Synthesis(),
        _Judge(),
        _Reranker(),
        _Embedding(),
        GlobalProcedureHubSettings(),
    )
    graph = GlobalSemanticGraph(
        node,
        GlobalProcedureHubPersistenceNode(_Repository()),
        node,
        GlobalProcedureHubPersistenceNode(_Repository()),
        node,
        GlobalProcedureHubPersistenceNode(_Repository()),
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
