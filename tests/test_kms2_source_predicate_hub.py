import asyncio

from kms2.config import SourcePredicateHubSettings
from kms2.core.model import (
    SourcePredicateHubCandidate,
    SourcePredicateHubDefinition,
    SourcePredicateHubJudgeInput,
    SourcePredicateHubMember,
)
from kms2.core.model.semantic.source_predicate_hub import (
    SourcePredicateHubJudgeDecision,
)
from kms2.database.semantic.queries import (
    DETECT_SOURCE_PREDICATE_COMMUNITIES,
    READ_SOURCE_PREDICATE_HUB_CANDIDATES,
    REPLACE_SOURCE_PREDICATE_ACCEPTED_EDGES,
)
from kms2.database.semantic.repository import SemanticRepository
from kms2.langgraph.semantic.state import SemanticState
from kms2.node.semantic.source_predicate_hub import SourcePredicateHubNode


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
        if query is READ_SOURCE_PREDICATE_HUB_CANDIDATES:
            return _Result(
                [
                    {
                        'left_uuid': 'predicate-1',
                        'left_predicate': 'supports',
                        'left_description': 'provides support',
                        'left_subject': 'Alice',
                        'left_object': 'Acme',
                        'right_uuid': 'predicate-2',
                        'right_predicate': 'supports',
                        'right_description': 'provides support',
                        'right_subject': 'Bob',
                        'right_object': 'Acme',
                        'score': 0.9,
                    }
                ]
            )
        if query is DETECT_SOURCE_PREDICATE_COMMUNITIES:
            return _Result(
                [
                    {
                        'community_id': 1,
                        'uuid': 'predicate-1',
                        'predicate': 'supports',
                        'description': 'provides support',
                    },
                    {
                        'community_id': 1,
                        'uuid': 'predicate-2',
                        'predicate': 'supports',
                        'description': 'provides support',
                    },
                ]
            )
        return _Result()


def _candidate():
    return SourcePredicateHubCandidate(
        left_uuid='predicate-1',
        left_predicate='supports',
        left_description='provides support',
        left_subject='Alice',
        left_object='Acme',
        right_uuid='predicate-2',
        right_predicate='supports',
        right_description='provides support',
        right_subject='Bob',
        right_object='Acme',
        score=0.9,
    )


def test_predicate_repository_includes_directed_endpoint_context():
    session = _Session()
    repository = SemanticRepository(lambda: _Context(session))

    candidates = asyncio.run(
        repository.read_source_predicate_hub_candidates(
            'source-1', candidate_limit=20, minimum_similarity=0.9
        )
    )
    asyncio.run(
        repository.replace_source_predicate_accepted_edges(
            'source-1', candidates
        )
    )

    assert candidates[0].left_subject == 'Alice'
    assert candidates[0].right_object == 'Acme'
    assert 'HAS_SUBJECT' in session.calls[0][0]
    assert 'HAS_OBJECT' in session.calls[0][0]
    assert session.calls[1][0] is REPLACE_SOURCE_PREDICATE_ACCEPTED_EDGES
    assert 'relevance_score' not in session.calls[1][1]['pairs'][0]


class _Repository:
    def __init__(self):
        self.accepted = None

    async def read_source_predicate_hub_candidates(self, *args, **kwargs):
        return [_candidate()]

    async def replace_source_predicate_accepted_edges(self, source_uuid, pairs):
        self.accepted = (source_uuid, pairs)

    async def detect_source_predicate_communities(self, *args, **kwargs):
        return [
            [
                SourcePredicateHubMember(
                    uuid='predicate-1',
                    predicate='supports',
                    description='provides support',
                ),
                SourcePredicateHubMember(
                    uuid='predicate-2',
                    predicate='supports',
                    description='provides support',
                ),
            ]
        ]


class _Synthesis:
    async def aforward(self, *, request):
        return SourcePredicateHubDefinition(
            predicate='supports', description='provides support'
        )


class _Embedding:
    async def embed(self, texts):
        assert list(texts) == ['supports: provides support']
        return [[0.7]]


class _Reranker:
    async def rerank(self, query, documents, top_n=None):
        return [{'index': 0, 'relevance_score': 0.5}]


class _Judge:
    def __init__(self):
        self.requests = []

    async def aforward(self, *, requests: list[SourcePredicateHubJudgeInput]):
        self.requests.append(requests)
        return [
            SourcePredicateHubJudgeDecision(
                index=index, belongs_in_same_hub=True
            )
            for index, _ in enumerate(requests)
        ]


async def _noop():
    return None


def test_predicate_node_judge_receives_both_directed_contexts():
    repository = _Repository()
    judge = _Judge()
    node = SourcePredicateHubNode(
        repository,
        _Synthesis(),
        judge,
        _Reranker(),
        _Embedding(),
        SourcePredicateHubSettings(),
        _noop,
    )

    asyncio.run(node.run(SemanticState(source_uuid='source-1')))

    request = judge.requests[0][0]
    assert request.left_subject == 'Alice'
    assert request.left_object == 'Acme'
    assert request.right_subject == 'Bob'
    assert request.right_object == 'Acme'
    assert [pair.right_uuid for pair in repository.accepted[1]] == [
        'predicate-2'
    ]


def test_predicate_node_limits_judge_batches():
    judge = _Judge()
    node = SourcePredicateHubNode(
        _Repository(),
        _Synthesis(),
        judge,
        _Reranker(),
        _Embedding(),
        SourcePredicateHubSettings(judge_batch_size=2),
        _noop,
    )

    asyncio.run(node._judge_borderline([_candidate() for _ in range(3)]))

    assert [len(requests) for requests in judge.requests] == [2, 1]
