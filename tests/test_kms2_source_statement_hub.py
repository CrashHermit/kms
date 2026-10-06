import asyncio

from kms2_token_helpers import (
    JUDGE_BUDGET,
    RERANKER_COUNTER,
    direct_synthesis,
)

from kms2.config.source_semantic import SourceStatementHubSettings
from kms2.core.model.source_semantic.source_statement_hub import (
    SourceStatementHubCandidate,
    SourceStatementHubDefinition,
    SourceStatementHubMember,
    SourceStatementHubSynthesisResult,
)
from kms2.langgraph.source_semantic.state import SourceSemanticState
from kms2.node.source_semantic.source_statement_hub import (
    SourceStatementHubNode,
)


class _Repository:
    async def read_source_statement_hub_candidates(self, *args, **kwargs):
        return []

    async def replace_source_statement_accepted_edges(self, *args):
        self.pairs = args[1]

    async def detect_source_statement_communities(self, *args, **kwargs):
        return []


class _Embedding:
    async def embed(self, texts):
        assert list(texts) == []


class _Synthesis:
    def __init__(self):
        self.requests = []

    async def acall(self, *, request):
        self.requests.append(request)
        return SourceStatementHubDefinition(
            canonical_name='Conservation law',
            description='Energy remains constant in a closed system.',
        )


async def _run_empty_statement(node, state):
    state = state.model_copy(update=await node.load_candidates(state))
    state = state.model_copy(update={'source_statement_hub_rerank_results': []})
    state = state.model_copy(update=node.collect_rerank(state))
    state = state.model_copy(update={'source_statement_hub_judge_results': []})
    state = state.model_copy(update=node.collect_judge(state))
    state = state.model_copy(update=await node.detect_communities(state))
    state = state.model_copy(
        update={'source_statement_hub_synthesis_results': []}
    )
    state = state.model_copy(update=node.collect_synthesis(state))
    return await node.embed(state)


def test_statement_hub_empty_source_is_independent_and_has_no_exercise_input():
    node = SourceStatementHubNode(
        _Repository(),
        None,
        None,
        None,
        _Embedding(),
        SourceStatementHubSettings(),
        reranker_token_counter=RERANKER_COUNTER,
        judge_budget=JUDGE_BUDGET,
        **direct_synthesis(),
    )
    result = asyncio.run(
        _run_empty_statement(node, SourceSemanticState(source_uuid='source-1'))
    )
    assert result == {
        'source_statement_hubs': [],
        'source_statement_hub_memberships': [],
    }


def test_statement_hub_candidate_contains_only_description_evidence():
    candidate = SourceStatementHubCandidate(
        left_uuid='a',
        left_description='claim A',
        right_uuid='b',
        right_description='claim B',
        score=0.9,
    )
    assert candidate.model_dump() == {
        'left_uuid': 'a',
        'left_description': 'claim A',
        'right_uuid': 'b',
        'right_description': 'claim B',
        'score': 0.9,
    }


def test_statement_synthesis_keeps_model_evidence_and_membership_separate():
    synthesis = _Synthesis()
    node = SourceStatementHubNode(
        None,
        synthesis,
        None,
        None,
        _Embedding(),
        SourceStatementHubSettings(),
        reranker_token_counter=RERANKER_COUNTER,
        judge_budget=JUDGE_BUDGET,
        **direct_synthesis(),
    )
    member = SourceStatementHubMember(
        uuid='statement-1',
        description='Energy remains constant in a closed system.',
    )
    result = asyncio.run(
        node.synthesis_worker(
            {
                'source_statement_hub_synthesis_ordinal': 4,
                'source_statement_hub_community': [member],
            }
        )
    )['source_statement_hub_synthesis_results'][0]

    assert synthesis.requests[0].members[0].description == member.description
    assert result.membership_uuids == ['statement-1']
    assert 'aliases' not in result.model_dump()


def test_statement_synthesis_result_contains_only_embedding_inputs():
    result = SourceStatementHubSynthesisResult(
        ordinal=0,
        definition=SourceStatementHubDefinition(
            canonical_name='Conservation law',
            description='Energy remains constant in a closed system.',
        ),
        membership_uuids=['statement-1'],
    )

    assert result.model_dump() == {
        'ordinal': 0,
        'definition': {
            'canonical_name': 'Conservation law',
            'description': 'Energy remains constant in a closed system.',
        },
        'membership_uuids': ['statement-1'],
    }
