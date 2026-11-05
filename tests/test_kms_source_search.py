"""Behavioral tests for source search ranking and evidence provenance."""

import asyncio
from pathlib import Path

from kms.config.source_search import SourceSearchSettings
from kms.core.content import ContentImage
from kms.core.model.block import SourceBlock
from kms.core.model.content import MultimodalContent
from kms.core.model.search import SearchMaterialKind
from kms.core.model.source import Source
from kms.core.model.source_evidence import (
    SourceEvidenceBlock,
    SourceEvidenceGroup,
    SourcePassage,
    SourceProcedureSearchCandidate,
    SourceSearchCandidate,
    SourceSearchHit,
    SourceStatementSearchCandidate,
)
from kms.core.model.source_semantic.source_procedure import SourceProcedure
from kms.core.model.source_semantic.source_statement import SourceStatement
from kms.core.model.visual_asset import VisualAsset
from kms.database.source.queries.source_blocks import (
    READ_OWNED_SOURCE_PROCEDURES,
    READ_OWNED_SOURCE_STATEMENTS,
)
from kms.database.source.source_block_repository import SourceBlockRepository
from kms.service.source_evidence import SourceSearchService


class _FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[tuple[list[MultimodalContent], list, str]] = []

    async def embed(self, materials, *, content_images, input_type):
        self.calls.append((materials, content_images, input_type))
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


class _FakeReranker:
    def __init__(self, results) -> None:
        self.results = results
        self.calls = []

    async def rerank(self, query, documents, *, content_images, top_n=None):
        self.calls.append((query, documents, content_images, top_n))
        return self.results


class _Selection:
    def __init__(self, positions=None):
        self.positions = positions
        self.calls = []

    async def select(self, *, query, material_kind, candidates, content_images):
        self.calls.append((query, material_kind, candidates, content_images))
        return (
            list(range(len(candidates)))
            if self.positions is None
            else self.positions
        )


def _candidate(
    block_uuid: str,
    *,
    content: str | None,
    similarity: float,
    assets: list[VisualAsset] | None = None,
) -> SourceSearchCandidate:
    return SourceSearchCandidate(
        source=Source(uuid=f'{block_uuid}-source', key=f'{block_uuid}.pdf'),
        block=SourceBlock(
            uuid=block_uuid,
            block_type='paragraph',
            content=content,
            assets=assets or [],
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
    embedder = _FakeEmbedder()
    repository = _SourceBlockRepository(candidates)
    reranker = _FakeReranker(
        [
            {'index': 2, 'relevance_score': 0.97},
            {'index': 0, 'relevance_score': 0.73},
        ]
    )
    selection = _Selection([1, 0])
    service = SourceSearchService(
        repository,
        embedder,
        reranker,
        SourceSearchSettings(candidate_limit=12, result_limit=2),
        selection=selection,
    )

    hits = asyncio.run(
        service.search_sources('user-1', 'query', source_uuids=['source-2'])
    )

    assert [hit.block.uuid for hit in hits] == ['block-a', 'block-c']
    assert [
        (
            hit.source.uuid,
            hit.page_indexes,
            hit.similarity,
            hit.relevance_score,
        )
        for hit in hits
    ] == [
        ('block-a-source', [2, 3], 0.92, 0.73),
        ('block-c-source', [2, 3], 0.81, 0.97),
    ]
    assert embedder.calls == [([MultimodalContent(text='query')], [], 'query')]
    assert repository.calls == [('user-1', [0.4, 0.6], 12, ['source-2'])]
    rerank_query, rerank_materials, rerank_images, rerank_limit = (
        reranker.calls[0]
    )
    assert rerank_query == MultimodalContent(text='query')
    assert [material.text for material in rerank_materials] == [
        'Candidate A',
        'Candidate B',
        'Candidate C',
    ]
    assert rerank_images == []
    assert rerank_limit == 2
    assert selection.calls[0][1] is SearchMaterialKind.SOURCE_BLOCK
    assert [candidate.material.text for candidate in selection.calls[0][2]] == [
        'Candidate C',
        'Candidate A',
    ]


def test_image_only_source_candidate_reaches_reranker_and_selector():
    asset = VisualAsset(uuid='asset-1', path='/assets/figure.png')
    candidate = _candidate(
        'image-only',
        content=None,
        similarity=0.8,
        assets=[asset],
    )
    reranker = _FakeReranker([{'index': 0, 'relevance_score': 0.9}])
    selection = _Selection()
    service = SourceSearchService(
        _SourceBlockRepository([candidate]),
        _FakeEmbedder(),
        reranker,
        SourceSearchSettings(),
        selection=selection,
    )

    hits = asyncio.run(service.search_sources('user-1', 'query'))

    assert [hit.block.uuid for hit in hits] == ['image-only']
    assert hits[0].block.content is None
    assert hits[0].source.uuid == candidate.source.uuid
    assert hits[0].page_indexes == [2, 3]
    query_material, documents, images, _top_n = reranker.calls[0]
    assert query_material == MultimodalContent(text='query')
    assert documents == [MultimodalContent(text=None, image_positions=[0])]
    assert images == [
        ContentImage(key=('path', asset.path), payload=Path(asset.path))
    ]
    selected = selection.calls[0]
    assert selected[3] == images
    assert selected[2][0].material == documents[0]


def test_search_returns_empty_candidates_without_reranking():
    embedder = _FakeEmbedder()
    repository = _SourceBlockRepository([])
    reranker = _FakeReranker([])
    selection = _Selection()
    service = SourceSearchService(
        repository,
        embedder,
        reranker,
        SourceSearchSettings(),
        selection=selection,
    )

    hits: list[SourceSearchHit] = asyncio.run(
        service.search_sources('user-1', 'query')
    )

    assert hits == []
    assert embedder.calls == [([MultimodalContent(text='query')], [], 'query')]
    assert reranker.calls == []
    assert selection.calls == []


def test_search_with_empty_source_allowlist_skips_inference_and_storage():
    embedder = _FakeEmbedder()
    repository = _SourceBlockRepository([])
    reranker = _FakeReranker([])
    selection = _Selection()
    service = SourceSearchService(
        repository,
        embedder,
        reranker,
        SourceSearchSettings(),
        selection=selection,
    )

    hits = asyncio.run(
        service.search_sources('user-1', 'query', source_uuids=[])
    )

    assert hits == []
    assert embedder.calls == []
    assert repository.calls == []
    assert reranker.calls == []
    assert selection.calls == []


def test_search_preserves_provenance_in_reranker_order():
    candidates = [
        _candidate('block-a', content='alpha', similarity=0.95),
        _candidate('block-b', content='bravo', similarity=0.90),
        _candidate('block-c', content='charlie', similarity=0.85),
    ]

    class Reranker:
        async def rerank(self, query, documents, *, content_images, top_n=None):
            scores = {'alpha': 0.3, 'bravo': 0.6, 'charlie': 0.9}
            results = sorted(
                (
                    {
                        'index': index,
                        'relevance_score': scores[document.text],
                    }
                    for index, document in enumerate(documents)
                ),
                key=lambda result: -result['relevance_score'],
            )
            return results if top_n is None else results[:top_n]

    service = SourceSearchService(
        _SourceBlockRepository(candidates),
        _FakeEmbedder(),
        Reranker(),
        SourceSearchSettings(candidate_limit=3, result_limit=2),
        selection=_Selection(),
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

        async def rerank(self, query, documents, *, content_images, top_n=None):
            self.calls.append((query, documents, content_images, top_n))
            return [
                {'index': 1, 'relevance_score': 0.95},
                {'index': 0, 'relevance_score': 0.65},
            ]

    repository = Repository()
    embedder = _FakeEmbedder()
    reranker = Reranker()
    selection = _Selection()
    service = SourceSearchService(
        repository,
        embedder,
        reranker,
        SourceSearchSettings(candidate_limit=16, result_limit=2),
        selection=selection,
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
    assert [
        (call[0], [material.text for material in call[1]], call[2], call[3])
        for call in reranker.calls
    ] == [
        (
            MultimodalContent(text='exercise'),
            ['exercise statement', 'definition statement'],
            [],
            2,
        ),
        (
            MultimodalContent(text='method'),
            ['procedure text A', 'procedure text B'],
            [],
            2,
        ),
    ]
    assert [candidate.is_exercise for candidate in selection.calls[0][2]] == [
        False,
        True,
    ]
    assert selection.calls[0][1] is SearchMaterialKind.STATEMENT
    assert selection.calls[1][1] is SearchMaterialKind.PROCEDURE
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
    assert embedder.calls == [
        ([MultimodalContent(text='exercise')], [], 'query'),
        ([MultimodalContent(text='method')], [], 'query'),
    ]


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
    embedder = _FakeEmbedder()
    repository = _SourceBlockRepository([])
    reranker = _FakeReranker([])
    selection = _Selection()
    service = SourceSearchService(
        repository,
        embedder,
        reranker,
        SourceSearchSettings(),
        selection=selection,
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
    assert embedder.calls == []
    assert repository.calls == []
    assert reranker.calls == []
    assert selection.calls == []
