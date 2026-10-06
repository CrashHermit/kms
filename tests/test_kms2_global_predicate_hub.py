import asyncio

from kms2_token_helpers import (
    JUDGE_BUDGET,
    RERANKER_COUNTER,
    direct_synthesis,
)

from kms2.config.global_semantic import GlobalPredicateHubSettings
from kms2.core.model.global_semantic.global_predicate_hub import (
    GlobalPredicateHub,
    GlobalPredicateHubCandidate,
    GlobalPredicateHubDefinition,
    GlobalPredicateHubJudgeDecision,
    GlobalPredicateHubJudgeInput,
    GlobalPredicateHubMember,
)
from kms2.langgraph.global_semantic.state import GlobalSemanticState
from kms2.node.global_semantic.global_predicate_hub import (
    GlobalPredicateHubNode,
)


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
    async def acall(self, *, request):
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

    async def acall(self, *, requests: list[GlobalPredicateHubJudgeInput]):
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
        **direct_synthesis(),
        reranker_token_counter=RERANKER_COUNTER,
        judge_budget=JUDGE_BUDGET,
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
