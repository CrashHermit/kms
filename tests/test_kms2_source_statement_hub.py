import asyncio

from kms2.config import SourceStatementHubSettings
from kms2.core.model import SourceStatementHubCandidate
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


async def _noop():
    return None


def test_statement_hub_empty_source_is_independent_and_has_no_exercise_input():
    node = SourceStatementHubNode(
        _Repository(),
        None,
        None,
        None,
        _Embedding(),
        SourceStatementHubSettings(),
        _noop,
    )
    result = asyncio.run(node.run(SemanticState(source_uuid='source-1')))
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
