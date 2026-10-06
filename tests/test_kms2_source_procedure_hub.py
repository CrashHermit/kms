import asyncio

from kms2_token_helpers import (
    JUDGE_BUDGET,
    RERANKER_COUNTER,
    direct_synthesis,
)

from kms2.config.source_semantic import SourceProcedureHubSettings
from kms2.core.model.source_semantic.source_procedure_hub import (
    SourceProcedureHubCandidate,
    SourceProcedureHubDefinition,
    SourceProcedureHubMember,
    SourceProcedureHubSynthesisResult,
)
from kms2.langgraph.source_semantic.state import SourceSemanticState
from kms2.node.source_semantic.source_procedure_hub import (
    SourceProcedureHubNode,
)


class _Repository:
    async def read_source_procedure_hub_candidates(self, *args, **kwargs):
        return []

    async def replace_source_procedure_accepted_edges(self, *args):
        self.pairs = args[1]

    async def detect_source_procedure_communities(self, *args, **kwargs):
        return []


class _Embedding:
    async def embed(self, texts):
        assert list(texts) == []
        return []


class _Synthesis:
    def __init__(self):
        self.requests = []

    async def acall(self, *, request):
        self.requests.append(request)
        return SourceProcedureHubDefinition(
            canonical_name='Original order',
            description='Apply the steps in order.',
        )


def test_procedure_worker_keeps_original_membership_outside_synthesis():
    module = _Synthesis()
    node = SourceProcedureHubNode(
        _Repository(),
        module,
        None,
        None,
        _Embedding(),
        SourceProcedureHubSettings(),
        reranker_token_counter=RERANKER_COUNTER,
        judge_budget=JUDGE_BUDGET,
        **direct_synthesis(),
    )
    community = [
        SourceProcedureHubMember(uuid='procedure-1', description='First step.'),
        SourceProcedureHubMember(
            uuid='procedure-2', description='Then finish.'
        ),
    ]

    result = asyncio.run(
        node.synthesis_worker(
            {
                'source_procedure_hub_synthesis_ordinal': 3,
                'source_procedure_hub_community': community,
            }
        )
    )

    assert [member.description for member in module.requests[0].members] == [
        'First step.',
        'Then finish.',
    ]
    result_item = result['source_procedure_hub_synthesis_results'][0]
    assert result_item.membership_uuids == ['procedure-1', 'procedure-2']


async def _run_empty_procedure(node, state):
    state = state.model_copy(update=await node.load_candidates(state))
    state = state.model_copy(update={'source_procedure_hub_rerank_results': []})
    state = state.model_copy(update=node.collect_rerank(state))
    state = state.model_copy(update={'source_procedure_hub_judge_results': []})
    state = state.model_copy(update=node.collect_judge(state))
    state = state.model_copy(update=await node.detect_communities(state))
    state = state.model_copy(
        update={'source_procedure_hub_synthesis_results': []}
    )
    state = state.model_copy(update=node.collect_synthesis(state))
    return await node.embed(state)


def test_procedure_hub_empty_source_is_independent():
    node = SourceProcedureHubNode(
        _Repository(),
        None,
        None,
        None,
        _Embedding(),
        SourceProcedureHubSettings(),
        reranker_token_counter=RERANKER_COUNTER,
        judge_budget=JUDGE_BUDGET,
        **direct_synthesis(),
    )
    result = asyncio.run(
        _run_empty_procedure(node, SourceSemanticState(source_uuid='source-1'))
    )
    assert result == {
        'source_procedure_hubs': [],
        'source_procedure_hub_memberships': [],
    }


def test_procedure_hub_candidate_preserves_ordered_description_evidence():
    candidate = SourceProcedureHubCandidate(
        left_uuid='a',
        left_description='step one then step two',
        right_uuid='b',
        right_description='step one then step two',
        score=0.9,
    )
    assert candidate.left_description == candidate.right_description


def test_procedure_synthesis_result_contains_only_embedding_inputs():
    result = SourceProcedureHubSynthesisResult(
        ordinal=0,
        definition=SourceProcedureHubDefinition(
            canonical_name='Differentiate a polynomial',
            description='Apply the power rule to each term.',
        ),
        membership_uuids=['procedure-1'],
    )

    assert result.model_dump() == {
        'ordinal': 0,
        'definition': {
            'canonical_name': 'Differentiate a polynomial',
            'description': 'Apply the power rule to each term.',
        },
        'membership_uuids': ['procedure-1'],
    }
