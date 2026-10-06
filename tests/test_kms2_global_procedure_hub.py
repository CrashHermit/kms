import asyncio

from kms2_token_helpers import (
    JUDGE_BUDGET,
    RERANKER_COUNTER,
    direct_synthesis,
)

from kms2.config.global_semantic import GlobalProcedureHubSettings
from kms2.core.model.global_semantic.global_procedure_hub import (
    GlobalProcedureHub,
    GlobalProcedureHubCandidate,
    GlobalProcedureHubDefinition,
    GlobalProcedureHubJudgeDecision,
    GlobalProcedureHubJudgeInput,
    GlobalProcedureHubMember,
)
from kms2.langgraph.global_semantic.state import GlobalSemanticState
from kms2.node.global_semantic.global_procedure_hub import (
    GlobalProcedureHubNode,
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

    async def acall(self, *, request):
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

    async def acall(self, *, requests: list[GlobalProcedureHubJudgeInput]):
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
        reranker_token_counter=RERANKER_COUNTER,
        judge_budget=JUDGE_BUDGET,
        **direct_synthesis(),
    )

    state = GlobalSemanticState(
        **asyncio.run(node.load_candidates(GlobalSemanticState()))
    )
    sends = asyncio.run(node.dispatch_rerank(state))
    rerank_result = asyncio.run(node.rerank_worker(sends[0].arg))
    state = state.model_copy(update=rerank_result)
    state = state.model_copy(update=node.collect_rerank(state))
    judge_sends = asyncio.run(node.dispatch_judge(state))
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
