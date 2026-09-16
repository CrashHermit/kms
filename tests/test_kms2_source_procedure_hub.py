import asyncio

from kms2.config import SourceProcedureHubSettings
from kms2.core.model import SourceProcedureHubCandidate
from kms2.langgraph.semantic.state import SemanticState
from kms2.node.semantic.source_procedure_hub import SourceProcedureHubNode


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


async def _noop():
    return None


def test_procedure_hub_empty_source_is_independent():
    node = SourceProcedureHubNode(
        _Repository(),
        None,
        None,
        None,
        _Embedding(),
        SourceProcedureHubSettings(),
        _noop,
    )
    result = asyncio.run(node.run(SemanticState(source_uuid='source-1')))
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
