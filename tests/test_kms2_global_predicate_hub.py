import asyncio

from kms2.config.global_semantic import GlobalPredicateHubSettings
from kms2.core.model.global_semantic.global_predicate_hub import (
    GlobalPredicateHub,
    GlobalPredicateHubCandidate,
    GlobalPredicateHubDefinition,
    GlobalPredicateHubJudgeDecision,
    GlobalPredicateHubJudgeInput,
    GlobalPredicateHubMember,
)
from kms2.database.global_semantic.global_predicate_hub_repository import (
    GlobalPredicateHubRepository,
)
from kms2.database.global_semantic.queries.global_predicate_hub import (
    DETECT_GLOBAL_PREDICATE_HUB_COMMUNITIES,
    READ_GLOBAL_PREDICATE_HUB_CANDIDATES,
    REPLACE_GLOBAL_PREDICATE_HUB_ACCEPTED_EDGES,
)
from kms2.langgraph.global_semantic.graph import GlobalSemanticGraph
from kms2.langgraph.global_semantic.state import GlobalSemanticState
from kms2.node.global_semantic.global_predicate_hub import (
    GlobalPredicateHubNode,
)
from kms2.node.global_semantic.global_predicate_hub_persistence import (
    GlobalPredicateHubPersistenceNode,
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
        if query is READ_GLOBAL_PREDICATE_HUB_CANDIDATES:
            return _Result(
                [
                    {
                        'left_uuid': 'hub-1',
                        'left_predicate': 'supports',
                        'left_description': 'provides support',
                        'right_uuid': 'hub-2',
                        'right_predicate': 'supports',
                        'right_description': 'provides support',
                        'score': 0.9,
                    }
                ]
            )
        if query is DETECT_GLOBAL_PREDICATE_HUB_COMMUNITIES:
            return _Result(
                [
                    {
                        'community_id': 1,
                        'uuid': 'hub-1',
                        'predicate': 'supports',
                        'description': 'provides support',
                    },
                    {
                        'community_id': 1,
                        'uuid': 'hub-2',
                        'predicate': 'supports',
                        'description': 'provides support',
                    },
                ]
            )
        return _Result()


def _candidate():
    return GlobalPredicateHubCandidate(
        left_uuid='hub-1',
        left_predicate='supports',
        left_description='provides support',
        right_uuid='hub-2',
        right_predicate='supports',
        right_description='provides support',
        score=0.9,
    )


def test_global_hub_model_uses_automatic_uuid_and_no_membership_property():
    hub = GlobalPredicateHub(
        predicate='supports',
        aliases=['supports'],
        description='provides support',
        embedding=[0.7],
    )

    assert hub.uuid
    assert 'source_uuid' not in GlobalPredicateHub.model_fields
    assert 'membership_uuids' not in GlobalPredicateHub.model_fields


def test_global_repository_uses_predicate_only_cross_source_queries():
    session = _Session()
    repository = GlobalPredicateHubRepository(lambda: _Context(session))

    candidates = asyncio.run(
        repository.read_global_predicate_hub_candidates(
            candidate_limit=20,
            minimum_similarity=0.9,
        )
    )
    asyncio.run(
        repository.replace_global_predicate_hub_accepted_edges(candidates)
    )
    communities = asyncio.run(
        repository.detect_global_predicate_hub_communities(
            max_iterations=10,
            min_association_strength=0.2,
            minimum_community_size=2,
        )
    )

    assert candidates == [_candidate()]
    assert len(communities) == 1
    assert 'HAS_SUBJECT' not in session.calls[0][0]
    assert 'HAS_OBJECT' not in session.calls[0][0]
    assert 'source_uuid' not in session.calls[0][1]
    assert 'candidate.source_uuid <> query.source_uuid' in session.calls[0][0]
    assert session.calls[1][0] is REPLACE_GLOBAL_PREDICATE_HUB_ACCEPTED_EDGES


class _Repository:
    def __init__(self):
        self.accepted = None

    async def read_global_predicate_hub_candidates(self, **kwargs):
        return [_candidate()]

    async def replace_global_predicate_hub_accepted_edges(self, pairs):
        self.accepted = pairs

    async def detect_global_predicate_hub_communities(self, **kwargs):
        return [
            [
                GlobalPredicateHubMember(
                    uuid='hub-1',
                    predicate='supports',
                    description='provides support',
                ),
                GlobalPredicateHubMember(
                    uuid='hub-2',
                    predicate='supports',
                    description='provides support',
                ),
            ]
        ]


class _Synthesis:
    async def aforward(self, *, request):
        return GlobalPredicateHubDefinition(
            predicate='supports', description='provides support'
        )


class _Embedding:
    async def embed(self, texts):
        assert list(texts) == ['supports: provides support']
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

    async def aforward(self, *, requests: list[GlobalPredicateHubJudgeInput]):
        self.requests.append(requests)
        return [
            GlobalPredicateHubJudgeDecision(
                index=index, belongs_in_same_hub=True
            )
            for index, _ in enumerate(requests)
        ]


def test_global_node_uses_only_canonical_predicate_evidence():
    repository = _Repository()
    reranker = _Reranker()
    judge = _Judge()
    node = GlobalPredicateHubNode(
        repository,
        _Synthesis(),
        judge,
        reranker,
        _Embedding(),
        GlobalPredicateHubSettings(),
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
    synthesis = node.dispatch_synthesis(state)
    result = asyncio.run(node.synthesis_worker(synthesis[0].arg))
    state = state.model_copy(update=result)
    state = state.model_copy(update=node.collect_synthesis(state))
    embedded = asyncio.run(node.embed(state))

    assert reranker.calls == [
        ('supports: provides support', ['supports: provides support'], None)
    ]
    request = judge.requests[0][0]
    assert request.left_predicate == 'supports'
    assert request.right_description == 'provides support'
    assert not hasattr(request, 'left_subject')
    assert repository.accepted == [_candidate()]
    assert embedded['global_predicate_hubs'][0].uuid
    assert embedded['global_predicate_hub_memberships'] == [['hub-1', 'hub-2']]


class _PhaseNode:
    def load_candidates(self, state):
        return {}

    def load_groups(self, state):
        return {}

    def run(self, state):
        return {}

    def dispatch_rerank(self, state):
        return 'global_predicate_hub_rerank_collect'

    def rerank_worker(self, state):
        return {}

    def collect_rerank(self, state):
        return {}

    def dispatch_judge(self, state):
        return 'global_predicate_hub_judge_collect'

    def judge_worker(self, state):
        return {}

    def collect_judge(self, state):
        return {}

    def detect_communities(self, state):
        return {}

    def dispatch_synthesis(self, state):
        return 'global_predicate_hub_synthesis_collect'

    def synthesis_worker(self, state):
        return {}

    def collect_synthesis(self, state):
        return {}

    def embed(self, state):
        return {}


class _Persistence:
    async def run(self, state):
        return {}


def test_global_graph_exposes_complete_phase_topology():
    node = GlobalPredicateHubNode(
        _Repository(),
        _Synthesis(),
        _Judge(),
        _Reranker(),
        _Embedding(),
        GlobalPredicateHubSettings(),
    )
    phase_node = _PhaseNode()
    persistence = _Persistence()
    graph = GlobalSemanticGraph(
        phase_node,
        persistence,
        phase_node,
        persistence,
        node,
        GlobalPredicateHubPersistenceNode(_Repository()),
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
        'global_event_hub_persistence',
        'global_predicate_hub_load',
        'global_predicate_hub_persistence',
        'global_statement_hub_load',
        'global_statement_hub_persistence',
        'global_procedure_hub_load',
        'global_procedure_hub_persistence',
    } <= set(graph.nodes)
