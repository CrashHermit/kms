import asyncio

from kms2.config import SourceEventHubSettings
from kms2.core.model import (
    SourceEventHubCandidate,
    SourceEventHubDefinition,
    SourceEventHubJudgeInput,
    SourceEventHubMember,
)
from kms2.core.model.semantic.source_event_hub import (
    SourceEventHubJudgeDecision,
)
from kms2.database.semantic.queries import (
    DETECT_SOURCE_EVENT_COMMUNITIES,
    READ_SOURCE_EVENT_HUB_CANDIDATES,
    REPLACE_SOURCE_EVENT_ACCEPTED_EDGES,
)
from kms2.database.semantic.repository import SemanticRepository
from kms2.langgraph.semantic.state import SemanticState
from kms2.node.semantic.source_event_hub import SourceEventHubNode


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
        if query is READ_SOURCE_EVENT_HUB_CANDIDATES:
            return _Result(
                [
                    {
                        'left_uuid': 'event-1',
                        'left_name': 'integration',
                        'left_description': 'the integration process',
                        'right_uuid': 'event-2',
                        'right_name': 'system integration',
                        'right_description': 'the same process',
                        'score': 0.88,
                    }
                ]
            )
        if query is DETECT_SOURCE_EVENT_COMMUNITIES:
            return _Result(
                [
                    {
                        'community_id': 1,
                        'uuid': 'event-1',
                        'name': 'integration',
                        'description': 'the integration process',
                    },
                    {
                        'community_id': 1,
                        'uuid': 'event-2',
                        'name': 'system integration',
                        'description': 'the same process',
                    },
                ]
            )
        return _Result()


def _candidate():
    return SourceEventHubCandidate(
        left_uuid='event-1',
        left_name='integration',
        left_description='the integration process',
        right_uuid='event-2',
        right_name='system integration',
        right_description='the same process',
        score=0.88,
    )


def test_event_repository_uses_typed_candidate_and_edge_queries():
    session = _Session()
    repository = SemanticRepository(lambda: _Context(session))

    candidates = asyncio.run(
        repository.read_source_event_hub_candidates(
            'source-1', candidate_limit=12, minimum_similarity=0.8
        )
    )
    asyncio.run(
        repository.replace_source_event_accepted_edges('source-1', candidates)
    )

    assert session.calls[0][0] is READ_SOURCE_EVENT_HUB_CANDIDATES
    assert 'source_event_embedding' in session.calls[0][0]
    assert 'candidate.source_uuid = $source_uuid' in session.calls[0][0]
    assert session.calls[1][0] is REPLACE_SOURCE_EVENT_ACCEPTED_EDGES
    assert 'similarity.score = pair.score' in session.calls[1][0]


class _Repository:
    def __init__(self):
        self.accepted = None

    async def read_source_event_hub_candidates(self, *args, **kwargs):
        return [_candidate()]

    async def replace_source_event_accepted_edges(self, source_uuid, pairs):
        self.accepted = (source_uuid, pairs)

    async def detect_source_event_communities(self, *args, **kwargs):
        return [
            [
                SourceEventHubMember(
                    uuid='event-1',
                    name='integration',
                    description='the integration process',
                ),
                SourceEventHubMember(
                    uuid='event-2',
                    name='system integration',
                    description='the same process',
                ),
            ]
        ]


class _Synthesis:
    async def aforward(self, *, request):
        return SourceEventHubDefinition(
            name='Integration', description='the integration process'
        )


class _Embedding:
    async def embed(self, texts):
        assert list(texts) == ['Integration: the integration process']
        return [[0.5]]


class _Reranker:
    async def rerank(self, query, documents, top_n=None):
        return [{'index': 0, 'relevance_score': 0.5}]


class _Judge:
    def __init__(self):
        self.requests = []

    async def aforward(self, *, requests: list[SourceEventHubJudgeInput]):
        self.requests.append(requests)
        return [
            SourceEventHubJudgeDecision(index=index, belongs_in_same_hub=False)
            for index, _ in enumerate(requests)
        ]


async def _noop():
    return None


def test_event_node_sends_borderline_pairs_only_to_event_judge():
    repository = _Repository()
    judge = _Judge()
    node = SourceEventHubNode(
        repository,
        _Synthesis(),
        judge,
        _Reranker(),
        _Embedding(),
        SourceEventHubSettings(),
        _noop,
    )

    result = asyncio.run(node.run(SemanticState(source_uuid='source-1')))

    assert len(judge.requests) == 1
    assert judge.requests[0][0].left_name == 'integration'
    assert repository.accepted[1] == []
    assert result['source_event_hubs'][0].name == 'Integration'


def test_event_node_limits_judge_batches():
    judge = _Judge()
    node = SourceEventHubNode(
        _Repository(),
        _Synthesis(),
        judge,
        _Reranker(),
        _Embedding(),
        SourceEventHubSettings(judge_batch_size=2),
        _noop,
    )

    asyncio.run(node._judge_borderline([_candidate() for _ in range(3)]))

    assert [len(requests) for requests in judge.requests] == [2, 1]
