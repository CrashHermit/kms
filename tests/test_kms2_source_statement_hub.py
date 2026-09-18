import asyncio

from kms2.config.semantic import SourceStatementHubSettings
from kms2.core.model import (
    SourceStatementHubCandidate,
    SourceStatementHubDefinition,
    SourceStatementHubSynthesisResult,
)
from kms2.langgraph.semantic.state import SemanticState
from kms2.node.semantic.source_statement_hub import SourceStatementHubNode


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
        return []


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
    )
    result = asyncio.run(
        _run_empty_statement(node, SemanticState(source_uuid='source-1'))
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
