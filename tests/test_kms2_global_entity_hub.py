import asyncio

from kms2.config.global_semantic import GlobalEntityHubSettings
from kms2.core.model.global_semantic.global_entity_hub import (
    GlobalEntityHub,
    GlobalEntityHubCandidate,
    GlobalEntityHubDefinition,
    GlobalEntityHubJudgeDecision,
    GlobalEntityHubJudgeInput,
    GlobalEntityHubMember,
)
from kms2.database.global_semantic.global_entity_hub_repository import (
    GlobalEntityHubRepository,
)
from kms2.database.global_semantic.queries.global_entity_hub import (
    DETECT_GLOBAL_ENTITY_HUB_COMMUNITIES,
    DROP_GLOBAL_ENTITY_HUB_GRAPH,
    READ_GLOBAL_ENTITY_HUB_CANDIDATES,
    REPLACE_GLOBAL_ENTITY_HUB_ACCEPTED_EDGES,
)
from kms2.langgraph.global_semantic.graph import GlobalSemanticGraph
from kms2.langgraph.global_semantic.state import GlobalSemanticState
from kms2.node.global_semantic.global_entity_hub import GlobalEntityHubNode
from kms2.node.global_semantic.global_entity_hub_persistence import (
    GlobalEntityHubPersistenceNode,
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
        if query is READ_GLOBAL_ENTITY_HUB_CANDIDATES:
            return _Result(
                [
                    {
                        'left_uuid': 'hub-1',
                        'left_canonical_name': 'Alice',
                        'left_description': 'one person',
                        'right_uuid': 'hub-2',
                        'right_canonical_name': 'Alice',
                        'right_description': 'one person',
                        'score': 0.9,
                    }
                ]
            )
        if query is DETECT_GLOBAL_ENTITY_HUB_COMMUNITIES:
            return _Result(
                [
                    {
                        'community_id': 1,
                        'uuid': 'hub-1',
                        'canonical_name': 'Alice',
                        'description': 'one person',
                    },
                    {
                        'community_id': 1,
                        'uuid': 'hub-2',
                        'canonical_name': 'Alicia',
                        'description': 'one person',
                    },
                ]
            )
        return _Result()


def _candidate():
    return GlobalEntityHubCandidate(
        left_uuid='hub-1',
        left_canonical_name='Alice',
        left_description='one person',
        right_uuid='hub-2',
        right_canonical_name='Alice',
        right_description='one person',
        score=0.9,
    )


def test_global_entity_hub_model_uses_automatic_uuid_and_no_membership_property():
    hub = GlobalEntityHub(
        canonical_name='Alice',
        aliases=['Alice'],
        description='one person',
        embedding=[0.7],
    )

    assert hub.uuid
    assert 'source_uuid' not in GlobalEntityHub.model_fields
    assert 'membership_uuids' not in GlobalEntityHub.model_fields


def test_global_entity_repository_uses_entity_only_cross_source_queries():
    session = _Session()
    repository = GlobalEntityHubRepository(lambda: _Context(session))

    candidates = asyncio.run(
        repository.read_global_entity_hub_candidates(
            candidate_limit=20,
            minimum_similarity=0.9,
        )
    )
    asyncio.run(repository.replace_global_entity_hub_accepted_edges(candidates))
    communities = asyncio.run(
        repository.detect_global_entity_hub_communities(
            max_iterations=10,
            min_association_strength=0.2,
            minimum_community_size=2,
        )
    )

    assert candidates == [_candidate()]
    assert len(communities) == 1
    assert session.calls[0][0] is READ_GLOBAL_ENTITY_HUB_CANDIDATES
    assert 'source_entity_hub_embedding' in session.calls[0][0]
    assert 'candidate.source_uuid <> query.source_uuid' in session.calls[0][0]
    assert 'HAS_SUBJECT' not in session.calls[0][0]
    assert 'HAS_OBJECT' not in session.calls[0][0]
    assert 'SourceEntity)' not in session.calls[0][0]
    assert 'source_uuid' not in session.calls[0][1]
    assert session.calls[1][0] is REPLACE_GLOBAL_ENTITY_HUB_ACCEPTED_EDGES
    assert session.calls[2][0] is DROP_GLOBAL_ENTITY_HUB_GRAPH
    assert session.calls[3][0] is DETECT_GLOBAL_ENTITY_HUB_COMMUNITIES
    assert session.calls[4][0] is DROP_GLOBAL_ENTITY_HUB_GRAPH


class _Repository:
    def __init__(self):
        self.accepted = None

    async def read_global_entity_hub_candidates(self, **kwargs):
        return [_candidate()]

    async def replace_global_entity_hub_accepted_edges(self, pairs):
        self.accepted = pairs

    async def detect_global_entity_hub_communities(self, **kwargs):
        return [
            [
                GlobalEntityHubMember(
                    uuid='hub-1',
                    canonical_name='Alice',
                    description='one person',
                ),
                GlobalEntityHubMember(
                    uuid='hub-2',
                    canonical_name='Alicia',
                    description='one person',
                ),
            ]
        ]


class _Synthesis:
    def __init__(self):
        self.requests = []

    async def aforward(self, *, request):
        self.requests.append(request)
        return GlobalEntityHubDefinition(
            canonical_name='Alice', description='one person'
        )


class _Embedding:
    async def embed(self, texts):
        assert list(texts) == ['Alice: one person']
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

    async def aforward(self, *, requests: list[GlobalEntityHubJudgeInput]):
        self.requests.append(requests)
        return [
            GlobalEntityHubJudgeDecision(index=index, belongs_in_same_hub=True)
            for index, _ in enumerate(requests)
        ]


def test_global_entity_node_uses_canonical_evidence_and_preserves_outputs():
    repository = _Repository()
    synthesis = _Synthesis()
    reranker = _Reranker()
    judge = _Judge()
    node = GlobalEntityHubNode(
        repository,
        synthesis,
        judge,
        reranker,
        _Embedding(),
        GlobalEntityHubSettings(),
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
        ('Alice: one person', ['Alice: one person'], None)
    ]
    request = judge.requests[0][0]
    assert request.left_canonical_name == 'Alice'
    assert request.right_description == 'one person'
    assert not hasattr(request, 'left_aliases')
    assert synthesis.requests[0].members[0].canonical_name == 'Alice'
    assert synthesis.requests[0].members[0].description == 'one person'
    assert repository.accepted == [_candidate()]
    assert embedded['global_entity_hubs'][0].uuid
    assert embedded['global_entity_hubs'][0].aliases == ['Alice', 'Alicia']
    assert embedded['global_entity_hub_memberships'] == [['hub-1', 'hub-2']]


class _PhaseNode:
    def load_candidates(self, state):
        return {}

    def load_groups(self, state):
        return {}

    def run(self, state):
        return {}

    def dispatch_rerank(self, state):
        return 'global_entity_hub_rerank_collect'

    def rerank_worker(self, state):
        return {}

    def collect_rerank(self, state):
        return {}

    def dispatch_judge(self, state):
        return 'global_entity_hub_judge_collect'

    def judge_worker(self, state):
        return {}

    def collect_judge(self, state):
        return {}

    def detect_communities(self, state):
        return {}

    def dispatch_synthesis(self, state):
        return 'global_entity_hub_synthesis_collect'

    def synthesis_worker(self, state):
        return {}

    def collect_synthesis(self, state):
        return {}

    def embed(self, state):
        return {}


class _Persistence:
    async def run(self, state):
        return {}


def test_global_graph_exposes_complete_entity_phase_topology():
    node = GlobalEntityHubNode(
        _Repository(),
        _Synthesis(),
        _Judge(),
        _Reranker(),
        _Embedding(),
        GlobalEntityHubSettings(),
    )
    phase_node = _PhaseNode()
    persistence = _Persistence()
    graph = GlobalSemanticGraph(
        node,
        GlobalEntityHubPersistenceNode(_Repository()),
        phase_node,
        persistence,
        phase_node,
        persistence,
        persistence,
        phase_node,
        persistence,
        phase_node,
        persistence,
        phase_node,
        persistence,
    ).build_graph()

    assert {
        'global_entity_hub_load',
        'global_entity_hub_rerank_worker',
        'global_entity_hub_rerank_collect',
        'global_entity_hub_judge_worker',
        'global_entity_hub_judge_collect',
        'global_entity_hub_communities',
        'global_entity_hub_synthesis_worker',
        'global_entity_hub_synthesis_collect',
        'global_entity_hub_embedding',
        'global_entity_hub_persistence',
        'global_event_hub_load',
        'global_predicate_hub_load',
        'global_predicate_hub_persistence',
        'global_triplet_projection',
        'global_triplet_hub_load',
        'global_triplet_hub_synthesis_worker',
        'global_triplet_hub_synthesis_collect',
        'global_triplet_hub_embedding',
        'global_triplet_hub_persistence',
        'global_statement_hub_load',
        'global_statement_hub_persistence',
        'global_procedure_hub_load',
        'global_procedure_hub_persistence',
    } <= set(graph.nodes)
