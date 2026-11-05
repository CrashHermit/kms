import asyncio

from kms_token_helpers import JUDGE_BUDGET, direct_synthesis

from kms.config.source_semantic import SourceProcedureHubSettings
from kms.core.model.source_semantic.source_procedure_hub import (
    SourceProcedureHubCandidate,
    SourceProcedureHubDefinition,
    SourceProcedureHubMember,
    SourceProcedureHubSynthesisResult,
)
from kms.langgraph.source_semantic.state import SourceSemanticState
from kms.node.source_semantic.source_procedure_hub import (
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
    async def embed(self, materials, *, content_images, input_type):
        assert list(materials) == []
        assert content_images == []
        assert input_type == 'document'
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
        judge_budget=JUDGE_BUDGET,
        **direct_synthesis(),
    )
    community = [
        SourceProcedureHubMember(uuid='procedure-1', content='First step.'),
        SourceProcedureHubMember(uuid='procedure-2', content='Then finish.'),
    ]

    result = asyncio.run(
        node.synthesis_worker(
            {
                'source_procedure_hub_synthesis_ordinal': 3,
                'source_procedure_hub_community': community,
            }
        )
    )

    assert [member.content for member in module.requests[0].members] == [
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


def test_procedure_hub_candidate_preserves_pair_ids_and_reconstructed_content():
    candidate = SourceProcedureHubCandidate(
        left_uuid='a',
        left_content='step one then step two',
        right_uuid='b',
        right_content='step one then step two',
        score=0.9,
    )
    assert (candidate.left_uuid, candidate.right_uuid) == ('a', 'b')
    assert candidate.left_content == candidate.right_content


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
