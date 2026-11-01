"""Behavioral tests for source search ranking and evidence provenance."""

import asyncio

from kms2.config.source_search import SourceSearchSettings
from kms2.core.model.block import SourceBlock
from kms2.core.model.source import Source
from kms2.core.model.source_evidence import (
    SourceEvidenceBlock,
    SourceEvidenceGroup,
    SourcePassage,
    SourceProcedureSearchCandidate,
    SourceSearchCandidate,
    SourceSearchHit,
    SourceStatementSearchCandidate,
)
from kms2.core.model.source_semantic.source_procedure import SourceProcedure
from kms2.core.model.source_semantic.source_statement import SourceStatement
from kms2.database.source.queries.source_blocks import (
    READ_OWNED_SOURCE_PROCEDURES,
    READ_OWNED_SOURCE_STATEMENTS,
)
from kms2.database.source.source_block_repository import SourceBlockRepository
from kms2.inference.reranking import BatchingRerankerClient
from kms2.service.source_evidence import SourceSearchService


class _EmbeddingClient:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[0.4, 0.6]]


class _SourceBlockRepository:
    def __init__(self, candidates: list[SourceSearchCandidate]) -> None:
        self.candidates = candidates
        self.calls: list[tuple[str, list[float], int, list[str] | None]] = []

    async def search_owned_blocks(
        self,
        user_uuid: str,
        query_embedding: list[float],
        *,
        candidate_limit: int,
        source_uuids: list[str] | None = None,
    ) -> list[SourceSearchCandidate]:
        self.calls.append(
            (user_uuid, query_embedding, candidate_limit, source_uuids)
        )
        return self.candidates


class _RerankerClient:
    def __init__(self, results) -> None:
        self.results = results
        self.calls: list[tuple[str, list[str], int | None]] = []

    async def rerank(self, query, documents, top_n=None):
        self.calls.append((query, documents, top_n))
        return self.results


def _candidate(
    block_uuid: str,
    *,
    content: str,
    similarity: float,
) -> SourceSearchCandidate:
    return SourceSearchCandidate(
        source=Source(uuid=f'{block_uuid}-source', key=f'{block_uuid}.pdf'),
        block=SourceBlock(
            uuid=block_uuid,
            block_type='paragraph',
            content=content,
        ),
        page_indexes=[2, 3],
        similarity=similarity,
    )


def test_search_keeps_reranker_order_scores_and_source_provenance():
    candidates = [
        _candidate('block-a', content='Candidate A', similarity=0.92),
        _candidate('block-b', content='Candidate B', similarity=0.88),
        _candidate('block-c', content='Candidate C', similarity=0.81),
    ]
    embedding_client = _EmbeddingClient()
    repository = _SourceBlockRepository(candidates)
    reranker_client = _RerankerClient(
        [
            {'index': 2, 'relevance_score': 0.97},
            {'index': 0, 'relevance_score': 0.73},
        ]
    )
    service = SourceSearchService(
        repository,
        embedding_client,
        reranker_client,
        SourceSearchSettings(candidate_limit=12, result_limit=2),
    )

    hits = asyncio.run(
        service.search_sources('user-1', 'query', source_uuids=['source-2'])
    )

    assert [hit.block.uuid for hit in hits] == ['block-c', 'block-a']
    assert [
        (
            hit.source.uuid,
            hit.page_indexes,
            hit.similarity,
            hit.relevance_score,
        )
        for hit in hits
    ] == [
        ('block-c-source', [2, 3], 0.81, 0.97),
        ('block-a-source', [2, 3], 0.92, 0.73),
    ]
    assert embedding_client.calls == [['query']]
    assert repository.calls == [('user-1', [0.4, 0.6], 12, ['source-2'])]
    assert reranker_client.calls == [
        ('query', ['Candidate A', 'Candidate B', 'Candidate C'], 2)
    ]


def test_search_returns_empty_candidates_without_reranking():
    embedding_client = _EmbeddingClient()
    repository = _SourceBlockRepository([])
    reranker_client = _RerankerClient([])
    service = SourceSearchService(
        repository,
        embedding_client,
        reranker_client,
        SourceSearchSettings(),
    )

    hits: list[SourceSearchHit] = asyncio.run(
        service.search_sources('user-1', 'query')
    )

    assert hits == []
    assert embedding_client.calls == [['query']]
    assert reranker_client.calls == []


def test_search_with_empty_source_allowlist_skips_inference_and_storage():
    embedding_client = _EmbeddingClient()
    repository = _SourceBlockRepository([])
    reranker_client = _RerankerClient([])
    service = SourceSearchService(
        repository,
        embedding_client,
        reranker_client,
        SourceSearchSettings(),
    )

    hits = asyncio.run(
        service.search_sources('user-1', 'query', source_uuids=[])
    )

    assert hits == []
    assert embedding_client.calls == []
    assert repository.calls == []
    assert reranker_client.calls == []


def test_search_ranks_and_preserves_provenance_across_reranker_batches():
    candidates = [
        _candidate('block-a', content='alpha', similarity=0.95),
        _candidate('block-b', content='bravo', similarity=0.90),
        _candidate('block-c', content='charlie', similarity=0.85),
    ]

    class Counter:
        def count_texts(self, texts):
            return [len(text) for text in texts]

    class Reranker:
        async def rerank(self, query, documents, top_n=None):
            scores = {'alpha': 0.3, 'bravo': 0.6, 'charlie': 0.9}
            return sorted(
                (
                    {'index': index, 'relevance_score': scores[document]}
                    for index, document in enumerate(documents)
                ),
                key=lambda result: -result['relevance_score'],
            )

    reranker = BatchingRerankerClient(
        Reranker(), Counter(), token_budget=10, max_concurrency=2
    )
    service = SourceSearchService(
        _SourceBlockRepository(candidates),
        _EmbeddingClient(),
        reranker,
        SourceSearchSettings(candidate_limit=3, result_limit=2),
    )
    hits = asyncio.run(service.search_sources('user-1', 'q'))

    assert [
        (hit.block.uuid, hit.source.uuid, hit.similarity, hit.relevance_score)
        for hit in hits
    ] == [
        ('block-c', 'block-c-source', 0.85, 0.9),
        ('block-b', 'block-b-source', 0.90, 0.6),
    ]


def test_typed_search_reranks_each_occurrence_type_with_its_own_content():
    statements = [
        SourceStatementSearchCandidate(
            statement=SourceStatement(
                uuid='statement-a',
                member_block_uuids=['block-a'],
                is_exercise=True,
                content='exercise statement',
                embedding=[0.1, 0.9],
            ),
            source=Source(uuid='source-a', key='algebra'),
            page_indexes=[1],
            similarity=0.7,
        ),
        SourceStatementSearchCandidate(
            statement=SourceStatement(
                uuid='statement-b',
                member_block_uuids=['block-b'],
                is_exercise=False,
                content='definition statement',
                embedding=[0.8, 0.2],
            ),
            source=Source(uuid='source-b', key='analysis'),
            page_indexes=[3, 4],
            similarity=0.9,
        ),
    ]
    procedures = [
        SourceProcedureSearchCandidate(
            procedure=SourceProcedure(
                uuid='procedure-a',
                member_block_uuids=['procedure-block-a'],
                content='procedure text A',
                embedding=[0.1, 0.9],
            ),
            source=Source(uuid='source-a', key='algebra'),
            page_indexes=[2],
            similarity=0.75,
        ),
        SourceProcedureSearchCandidate(
            procedure=SourceProcedure(
                uuid='procedure-b',
                member_block_uuids=['procedure-block-b'],
                content='procedure text B',
                embedding=[0.8, 0.2],
            ),
            source=Source(uuid='source-b', key='analysis'),
            page_indexes=[5],
            similarity=0.85,
        ),
    ]

    class Repository:
        def __init__(self):
            self.calls = []

        async def search_owned_statements(
            self,
            user_uuid,
            query_embedding,
            *,
            candidate_limit,
            source_uuids=None,
            is_exercise=None,
        ):
            self.calls.append(
                (
                    'statements',
                    user_uuid,
                    query_embedding,
                    candidate_limit,
                    source_uuids,
                    is_exercise,
                )
            )
            return statements

        async def search_owned_procedures(
            self,
            user_uuid,
            query_embedding,
            *,
            candidate_limit,
            source_uuids=None,
        ):
            self.calls.append(
                (
                    'procedures',
                    user_uuid,
                    query_embedding,
                    candidate_limit,
                    source_uuids,
                )
            )
            return procedures

    class Reranker:
        def __init__(self):
            self.calls = []

        async def rerank(self, query, documents, top_n=None):
            self.calls.append((query, documents, top_n))
            return [
                {'index': 1, 'relevance_score': 0.95},
                {'index': 0, 'relevance_score': 0.65},
            ]

    repository = Repository()
    embedding_client = _EmbeddingClient()
    reranker = Reranker()
    service = SourceSearchService(
        repository,
        embedding_client,
        reranker,
        SourceSearchSettings(candidate_limit=16, result_limit=2),
    )
    statement_hits = asyncio.run(
        service.search_statements(
            'user-1',
            'exercise',
            source_uuids=['source-a'],
            is_exercise=True,
        )
    )
    procedure_hits = asyncio.run(
        service.search_procedures(
            'user-1',
            'method',
            source_uuids=['source-b'],
        )
    )

    assert [
        (
            hit.statement.uuid,
            hit.source.uuid,
            hit.page_indexes,
            hit.similarity,
            hit.relevance_score,
        )
        for hit in statement_hits
    ] == [
        ('statement-b', 'source-b', [3, 4], 0.9, 0.95),
        ('statement-a', 'source-a', [1], 0.7, 0.65),
    ]
    assert [
        (
            hit.procedure.uuid,
            hit.source.uuid,
            hit.page_indexes,
            hit.similarity,
            hit.relevance_score,
        )
        for hit in procedure_hits
    ] == [
        ('procedure-b', 'source-b', [5], 0.85, 0.95),
        ('procedure-a', 'source-a', [2], 0.75, 0.65),
    ]
    assert reranker.calls == [
        (
            'exercise',
            ['exercise statement', 'definition statement'],
            2,
        ),
        ('method', ['procedure text A', 'procedure text B'], 2),
    ]
    assert repository.calls == [
        (
            'statements',
            'user-1',
            [0.4, 0.6],
            16,
            ['source-a'],
            True,
        ),
        ('procedures', 'user-1', [0.4, 0.6], 16, ['source-b']),
    ]
    assert embedding_client.calls == [['exercise'], ['method']]


def test_owned_occurrence_reads_preserve_source_revision_and_member_order():

    source_record = {
        'uuid': 'source-1',
        'key': 'textbook',
        'metadata': {},
        'revision': 4,
    }
    statement = {
        'uuid': 'statement-1',
        'is_exercise': True,
        'content': 'Exercise content',
        'embedding': [0.2, 0.8],
        'member_block_uuids': ['block-2', 'block-1'],
    }
    procedure = {
        'uuid': 'procedure-1',
        'content': 'Procedure content',
        'embedding': [0.3, 0.7],
        'member_block_uuids': ['block-3', 'block-4'],
    }
    queued_rows = [
        [{'source': source_record, 'statement': statement}],
        [{'source': source_record, 'procedure': procedure}],
    ]
    calls = []

    class Result:
        def __init__(self, rows):
            self.rows = rows

        async def data(self):
            return self.rows

        async def single(self, strict=False):
            assert strict is True
            assert len(self.rows) == 1
            return self.rows[0]

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exception_type, exception, traceback):
            return None

        async def run(self, query, **parameters):
            calls.append((query, parameters))
            return Result(queued_rows.pop(0))

    class Repository(SourceBlockRepository):
        async def read_passages(self, user_uuid, block_uuids, **kwargs):
            source = Source(**source_record)
            evidence_blocks = [
                SourceEvidenceBlock(
                    block=SourceBlock(
                        uuid=block_uuid,
                        block_type='paragraph',
                        content=block_uuid,
                    ),
                    page_indexes=[index],
                )
                for index, block_uuid in enumerate(block_uuids)
            ]
            return [
                SourcePassage(
                    source=source,
                    selected_block_uuids=block_uuids,
                    blocks=evidence_blocks,
                    groups=[
                        SourceEvidenceGroup(
                            uuid='group-1',
                            kind='statement',
                            member_block_uuids=block_uuids,
                        )
                    ],
                )
            ]

    repository = Repository(lambda: Session())
    statement_reads = asyncio.run(
        repository.read_statements('user-1', ['statement-1'])
    )
    procedure_reads = asyncio.run(
        repository.read_procedures('user-1', ['procedure-1'])
    )

    assert statement_reads[0].statement.is_exercise is True
    assert statement_reads[0].statement.member_block_uuids == [
        'block-2',
        'block-1',
    ]
    assert statement_reads[0].passages[0].source.revision == 4
    assert statement_reads[0].passages[0].selected_block_uuids == [
        'block-2',
        'block-1',
    ]
    assert procedure_reads[0].procedure.member_block_uuids == [
        'block-3',
        'block-4',
    ]
    assert procedure_reads[0].passages[0].source.revision == 4
    assert [query for query, _parameters in calls] == [
        READ_OWNED_SOURCE_STATEMENTS,
        READ_OWNED_SOURCE_PROCEDURES,
    ]


def test_typed_search_empty_allowlists_skip_embeddings_and_repository_calls():
    embedding_client = _EmbeddingClient()
    repository = _SourceBlockRepository([])
    reranker = _RerankerClient([])
    service = SourceSearchService(
        repository,
        embedding_client,
        reranker,
        SourceSearchSettings(),
    )

    assert (
        asyncio.run(
            service.search_statements(
                'user-1',
                'query',
                source_uuids=[],
                is_exercise=False,
            )
        )
        == []
    )
    assert (
        asyncio.run(
            service.search_procedures(
                'user-1',
                'query',
                source_uuids=[],
            )
        )
        == []
    )
    assert embedding_client.calls == []
    assert repository.calls == []
    assert reranker.calls == []
