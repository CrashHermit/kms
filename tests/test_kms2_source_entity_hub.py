import asyncio

from kms2.config import SourceEntityHubSettings
from kms2.core.model import (
    SourceEntityHubCandidate,
    SourceEntityHubDefinition,
    SourceEntityHubJudgeInput,
    SourceEntityHubMember,
)
from kms2.core.model.semantic.source_entity_hub import (
    SourceEntityHubJudgeDecision,
)
from kms2.database.semantic.queries import (
    DETECT_SOURCE_ENTITY_COMMUNITIES,
    DROP_SOURCE_ENTITY_HUB_GRAPH,
    READ_SOURCE_ENTITY_HUB_CANDIDATES,
    REPLACE_SOURCE_ENTITY_ACCEPTED_EDGES,
)
from kms2.database.semantic.repository import SemanticRepository
from kms2.langgraph.semantic.state import SemanticState
from kms2.node.semantic.source_entity_hub import SourceEntityHubNode


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
        if query is READ_SOURCE_ENTITY_HUB_CANDIDATES:
            return _Result(
                [
                    {
                        'left_uuid': 'entity-1',
                        'left_name': 'Alice',
                        'left_description': 'the person',
                        'right_uuid': 'entity-2',
                        'right_name': 'A. Smith',
                        'right_description': 'the same person',
                        'score': 0.91,
                    }
                ]
            )
        if query is DETECT_SOURCE_ENTITY_COMMUNITIES:
            return _Result(
                [
                    {
                        'community_id': 1,
                        'uuid': 'entity-1',
                        'name': 'Alice',
                        'description': 'the person',
                    },
                    {
                        'community_id': 1,
                        'uuid': 'entity-2',
                        'name': 'A. Smith',
                        'description': 'the same person',
                    },
                ]
            )
        return _Result()


def _candidate(left_uuid='entity-1', right_uuid='entity-2'):
    return SourceEntityHubCandidate(
        left_uuid=left_uuid,
        left_name='Alice',
        left_description='the person',
        right_uuid=right_uuid,
        right_name='A. Smith',
        right_description='the same person',
        score=0.91,
    )


def test_entity_repository_reads_candidates_and_replaces_only_accepted_edges():
    session = _Session()
    repository = SemanticRepository(lambda: _Context(session))

    candidates = asyncio.run(
        repository.read_source_entity_hub_candidates(
            'source-1', candidate_limit=17, minimum_similarity=0.82
        )
    )
    asyncio.run(
        repository.replace_source_entity_accepted_edges('source-1', candidates)
    )

    assert candidates[0].left_uuid == 'entity-1'
    assert session.calls[0][0] is READ_SOURCE_ENTITY_HUB_CANDIDATES
    assert session.calls[0][1] == {
        'source_uuid': 'source-1',
        'candidate_limit': 17,
        'minimum_similarity': 0.82,
    }
    assert session.calls[1][0] is REPLACE_SOURCE_ENTITY_ACCEPTED_EDGES
    assert session.calls[1][1]['pairs'][0]['score'] == 0.91
    assert 'relevance_score' not in session.calls[1][1]['pairs'][0]


def test_entity_community_detection_is_weighted_and_temporary():
    session = _Session()
    repository = SemanticRepository(lambda: _Context(session))

    communities = asyncio.run(
        repository.detect_source_entity_communities(
            'source-1',
            max_iterations=100,
            min_association_strength=0.2,
            minimum_community_size=2,
        )
    )

    assert [[member.uuid for member in group] for group in communities] == [
        ['entity-1', 'entity-2']
    ]
    assert session.calls[0][0] is DROP_SOURCE_ENTITY_HUB_GRAPH
    assert session.calls[1][0] is DETECT_SOURCE_ENTITY_COMMUNITIES
    assert "properties: 'score'" in session.calls[1][0]
    assert session.calls[2][0] is DROP_SOURCE_ENTITY_HUB_GRAPH


class _Repository:
    def __init__(self, candidates):
        self.candidates = candidates
        self.accepted = None

    async def read_source_entity_hub_candidates(self, *args, **kwargs):
        return self.candidates

    async def replace_source_entity_accepted_edges(self, source_uuid, pairs):
        self.accepted = (source_uuid, pairs)

    async def detect_source_entity_communities(self, *args, **kwargs):
        return [
            [
                SourceEntityHubMember(
                    uuid='entity-1', name='Alice', description='the person'
                ),
                SourceEntityHubMember(
                    uuid='entity-2',
                    name='A. Smith',
                    description='the same person',
                ),
            ]
        ]


class _Synthesis:
    async def aforward(self, *, request):
        assert request.members[0].description == 'the person'
        return SourceEntityHubDefinition(
            canonical_name='Alice', description='one person'
        )


class _Embedding:
    async def embed(self, texts):
        assert list(texts) == ['Alice: one person']
        return [[0.1]]


class _Reranker:
    def __init__(self):
        self.calls = []

    async def rerank(self, query, documents, top_n=None):
        self.calls.append((query, list(documents), top_n))
        return [
            {'index': index, 'relevance_score': score}
            for index, score in enumerate((1.0, 0.1, 0.5)[: len(documents)])
        ]


class _Judge:
    def __init__(self):
        self.requests = []

    async def aforward(self, *, requests: list[SourceEntityHubJudgeInput]):
        self.requests.append(requests)
        return [
            SourceEntityHubJudgeDecision(index=index, belongs_in_same_hub=True)
            for index, _ in enumerate(requests)
        ]


async def _noop():
    return None


def test_entity_node_routes_scores_and_stages_without_durable_hub_write():
    candidates = [_candidate() for _ in range(3)]
    candidates[1] = _candidate(right_uuid='entity-3')
    candidates[2] = _candidate(right_uuid='entity-4')
    repository = _Repository(candidates)
    reranker = _Reranker()
    judge = _Judge()
    node = SourceEntityHubNode(
        repository,
        _Synthesis(),
        judge,
        reranker,
        _Embedding(),
        SourceEntityHubSettings(),
        _noop,
    )

    result = asyncio.run(node.run(SemanticState(source_uuid='source-1')))

    assert reranker.calls[0][2] is None
    assert [pair.right_uuid for pair in repository.accepted[1]] == [
        'entity-2',
        'entity-4',
    ]
    assert len(judge.requests) == 1
    assert result['source_entity_hub_memberships'] == [['entity-1', 'entity-2']]


def test_entity_node_limits_judge_batches():
    judge = _Judge()
    node = SourceEntityHubNode(
        _Repository([]),
        _Synthesis(),
        judge,
        _Reranker(),
        _Embedding(),
        SourceEntityHubSettings(judge_batch_size=2),
        _noop,
    )

    asyncio.run(
        node._judge_borderline(
            [_candidate(right_uuid=f'entity-{index}') for index in range(3)]
        )
    )

    assert [len(requests) for requests in judge.requests] == [2, 1]
