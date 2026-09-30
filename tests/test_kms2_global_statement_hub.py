import asyncio

from langgraph.graph import END, START, StateGraph

from kms2.config.global_semantic import GlobalStatementHubSettings
from kms2.core.model.global_semantic.global_statement_hub import (
    GlobalStatementHub,
    GlobalStatementHubCandidate,
    GlobalStatementHubDefinition,
    GlobalStatementHubJudgeDecision,
    GlobalStatementHubJudgeInput,
    GlobalStatementHubMember,
    GlobalStatementHubSynthesisInput,
)
from kms2.langgraph.global_semantic.global_statement_hub import (
    add_global_statement_hub_phase,
)
from kms2.langgraph.global_semantic.state import GlobalSemanticState
from kms2.node.global_semantic.global_statement_hub import (
    GlobalStatementHubNode,
)
from kms2.node.global_semantic.global_statement_hub_persistence import (
    GlobalStatementHubPersistenceNode,
)


def _candidate():
    return GlobalStatementHubCandidate(
        left_uuid='hub-1',
        left_description='energy is conserved',
        right_uuid='hub-2',
        right_description='energy is conserved',
        score=0.9,
    )


def test_global_statement_hub_model_uses_automatic_uuid_and_no_membership_property():
    hub = GlobalStatementHub(
        canonical_name='conservation of energy',
        aliases=['energy conservation'],
        description='energy is conserved',
        embedding=[0.7],
    )

    assert hub.uuid
    assert 'source_uuid' not in GlobalStatementHub.model_fields
    assert 'membership_uuids' not in GlobalStatementHub.model_fields


class _Repository:
    def __init__(self):
        self.accepted = None

    async def read_global_statement_hub_candidates(self, **kwargs):
        return [_candidate()]

    async def replace_global_statement_hub_accepted_edges(self, pairs):
        self.accepted = pairs

    async def detect_global_statement_hub_communities(self, **kwargs):
        return [
            [
                GlobalStatementHubMember(
                    uuid='hub-1',
                    canonical_name='energy conservation',
                    description='energy is conserved',
                ),
                GlobalStatementHubMember(
                    uuid='hub-2',
                    canonical_name='conservation of energy',
                    description='energy is conserved',
                ),
            ]
        ]


class _Synthesis:
    def __init__(self):
        self.requests = []

    async def aforward(self, *, request):
        self.requests.append(request)
        return GlobalStatementHubDefinition(
            canonical_name='conservation of energy',
            description='energy remains constant in a closed system',
        )


class _Embedding:
    async def embed(self, texts):
        assert list(texts) == ['energy remains constant in a closed system']
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

    async def aforward(self, *, requests: list[GlobalStatementHubJudgeInput]):
        self.requests.append(requests)
        return [
            GlobalStatementHubJudgeDecision(
                index=index, belongs_in_same_hub=True
            )
            for index, _ in enumerate(requests)
        ]


def test_global_statement_node_uses_description_only_evidence_and_outputs_aliases():
    repository = _Repository()
    synthesis = _Synthesis()
    reranker = _Reranker()
    judge = _Judge()
    node = GlobalStatementHubNode(
        repository,
        synthesis,
        judge,
        reranker,
        _Embedding(),
        GlobalStatementHubSettings(),
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
        (
            'energy is conserved',
            ['energy is conserved'],
            None,
        )
    ]
    request = judge.requests[0][0]
    assert request.left_description == 'energy is conserved'
    assert request.right_description == 'energy is conserved'
    assert not hasattr(request, 'left_canonical_name')
    assert synthesis.requests == [
        GlobalStatementHubSynthesisInput(
            members=[
                {'description': 'energy is conserved'},
                {'description': 'energy is conserved'},
            ]
        )
    ]
    assert repository.accepted == [_candidate()]
    assert embedded['global_statement_hubs'][0].uuid
    assert embedded['global_statement_hubs'][0].canonical_name == (
        'conservation of energy'
    )
    assert embedded['global_statement_hubs'][0].aliases == [
        'energy is conserved',
        'energy is conserved',
    ]
    assert embedded['global_statement_hub_memberships'] == [['hub-1', 'hub-2']]


def test_global_statement_phase_exposes_complete_topology():
    repository = _Repository()
    node = GlobalStatementHubNode(
        repository,
        _Synthesis(),
        _Judge(),
        _Reranker(),
        _Embedding(),
        GlobalStatementHubSettings(),
    )
    graph = StateGraph(GlobalSemanticState)
    add_global_statement_hub_phase(
        graph,
        node,
        GlobalStatementHubPersistenceNode(repository),
    )
    graph.add_edge(START, 'global_statement_hub_load')
    graph.add_edge('global_statement_hub_persistence', END)
    compiled = graph.compile()

    assert {
        'global_statement_hub_load',
        'global_statement_hub_rerank_worker',
        'global_statement_hub_rerank_collect',
        'global_statement_hub_judge_worker',
        'global_statement_hub_judge_collect',
        'global_statement_hub_communities',
        'global_statement_hub_synthesis_worker',
        'global_statement_hub_synthesis_collect',
        'global_statement_hub_embedding',
        'global_statement_hub_persistence',
    } <= set(compiled.nodes)
