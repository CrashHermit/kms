"""Behavioral tests for source search selection and evidence provenance."""

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

from kms.core.embedding import EmbeddingRequest
from kms.media.models import ImageAsset, Material, TextPart
from kms.source.models.material import SourceProcedure, SourceStatement
from kms.source.models.source import Source, SourceBlock
from kms.source.retrieval.models import (
    SourceEvidenceBlock,
    SourceEvidenceGroup,
    SourcePassage,
    SourceProcedureSearchHit,
    SourceSearchHit,
    SourceStatementSearchHit,
)
from kms.source.retrieval.search import SourceSearchService
from kms.source.retrieval.selection.models import SearchMaterialKind
from kms.source.retrieval.settings import SourceSearchSettings
from kms.source.storage.retrieval import SourceRetrievalRepository
from kms.source.storage.retrieval_queries import (
    READ_OWNED_SOURCE_PROCEDURES,
    READ_OWNED_SOURCE_STATEMENTS,
)


class _FakeEmbedder:
    def __init__(self) -> None:
        self.calls = []

    async def embed(self, request: EmbeddingRequest):
        self.calls.append((request.materials, request.input_type))
        return [[0.4, 0.6]]


class _SourceRetrievalRepository:
    def __init__(self, candidates: list[SourceSearchHit]) -> None:
        self.candidates = candidates
        self.calls: list[tuple[str, list[float], int, list[str] | None]] = []

    async def search_owned_blocks(
        self,
        user_uuid: str,
        query_embedding: list[float],
        *,
        candidate_limit: int,
        source_uuids: list[str] | None = None,
    ) -> list[SourceSearchHit]:
        self.calls.append(
            (user_uuid, query_embedding, candidate_limit, source_uuids)
        )
        return self.candidates


class _Selection:
    def __init__(self, positions=None):
        self.positions = positions
        self.calls = []

    async def select(self, *, query, material_kind, candidates):
        self.calls.append((query, material_kind, candidates))
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
    image_path: str | None = None,
) -> SourceSearchHit:
    return SourceSearchHit(
        source=Source(uuid=f'{block_uuid}-source', key=f'{block_uuid}.pdf'),
        block=SourceBlock(
            uuid=block_uuid,
            block_type='image' if image_path is not None else 'paragraph',
            content=content,
            image_path=image_path,
            image_page_indexes=[2] if image_path is not None else [],
        ),
        page_indexes=[2, 3],
        similarity=similarity,
    )


def test_selector_gets_all_vector_candidates_and_caps_final_usefulness_order():
    candidates = [
        _candidate('block-a', content='Candidate A', similarity=0.92),
        _candidate('block-b', content='Candidate B', similarity=0.88),
        _candidate('block-c', content='Candidate C', similarity=0.81),
    ]
    embedder = _FakeEmbedder()
    repository = _SourceRetrievalRepository(candidates)
    selection = _Selection([2, 0, 1])
    service = SourceSearchService(
        repository,
        embedder,
        SourceSearchSettings(candidate_limit=12, result_limit=2),
        selection=selection,
    )

    hits = asyncio.run(
        service.search_sources('user-1', 'query', source_uuids=['source-2'])
    )

    assert hits == [candidates[2], candidates[0]]
    assert [hit.similarity for hit in hits] == [0.81, 0.92]
    assert embedder.calls == [([Material.from_text('query')], 'query')]
    assert repository.calls == [('user-1', [0.4, 0.6], 12, ['source-2'])]
    assert selection.calls[0][1] is SearchMaterialKind.SOURCE_BLOCK
    assert [item.material.text for item in selection.calls[0][2]] == [
        'Candidate A',
        'Candidate B',
        'Candidate C',
    ]
    assert [item.position for item in selection.calls[0][2]] == [0, 1, 2]


def test_image_only_source_candidate_reaches_selector():
    image_path = '/assets/figure.png'
    candidate = _candidate(
        'image-only',
        content=None,
        similarity=0.8,
        image_path=image_path,
    )
    selection = _Selection()
    service = SourceSearchService(
        _SourceRetrievalRepository([candidate]),
        _FakeEmbedder(),
        SourceSearchSettings(),
        selection=selection,
    )

    hits = asyncio.run(service.search_sources('user-1', 'query'))

    assert hits == [candidate]
    assert hits[0].block.content is None
    assert hits[0].source.uuid == candidate.source.uuid
    assert hits[0].page_indexes == [2, 3]
    selected = selection.calls[0]
    assert selected[2][0].material.images == (
        ImageAsset(key=('path', image_path), payload=Path(image_path)),
    )


def test_search_returns_empty_candidates_without_selection():
    embedder = _FakeEmbedder()
    repository = _SourceRetrievalRepository([])
    selection = _Selection()
    service = SourceSearchService(
        repository,
        embedder,
        SourceSearchSettings(),
        selection=selection,
    )

    hits: list[SourceSearchHit] = asyncio.run(
        service.search_sources('user-1', 'query')
    )

    assert hits == []
    assert embedder.calls == [([Material.from_text('query')], 'query')]
    assert selection.calls == []


def test_search_with_empty_source_allowlist_skips_inference_and_storage():
    embedder = _FakeEmbedder()
    repository = _SourceRetrievalRepository([])
    selection = _Selection()
    service = SourceSearchService(
        repository,
        embedder,
        SourceSearchSettings(),
        selection=selection,
    )

    hits = asyncio.run(
        service.search_sources('user-1', 'query', source_uuids=[])
    )

    assert hits == []
    assert embedder.calls == []
    assert repository.calls == []
    assert selection.calls == []


def test_search_preserves_provenance_when_selector_reorders_candidates():
    candidates = [
        _candidate('block-a', content='alpha', similarity=0.95),
        _candidate('block-b', content='bravo', similarity=0.90),
        _candidate('block-c', content='charlie', similarity=0.85),
    ]
    service = SourceSearchService(
        _SourceRetrievalRepository(candidates),
        _FakeEmbedder(),
        SourceSearchSettings(candidate_limit=3, result_limit=2),
        selection=_Selection([2, 1]),
    )

    hits = asyncio.run(service.search_sources('user-1', 'q'))

    assert [
        (hit.block.uuid, hit.source.uuid, hit.similarity) for hit in hits
    ] == [
        ('block-c', 'block-c-source', 0.85),
        ('block-b', 'block-b-source', 0.90),
    ]


def test_typed_search_preserves_vector_candidates_and_selector_usefulness_order():
    statements = [
        SourceStatementSearchHit(
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
        SourceStatementSearchHit(
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
        SourceProcedureSearchHit(
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
        SourceProcedureSearchHit(
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

        async def read_passages(
            self, user_uuid, block_uuids, *, expand_memberships, before, after
        ):
            assert user_uuid == 'user-1'
            assert expand_memberships and before == after == 0
            all_candidates = [*statements, *procedures]
            passages = []
            for source_uuid in ('source-a', 'source-b'):
                matching = [
                    candidate
                    for candidate in all_candidates
                    if candidate.source.uuid == source_uuid
                ]
                member_ids = [
                    uuid
                    for candidate in matching
                    for uuid in (
                        candidate.statement.member_block_uuids
                        if isinstance(candidate, SourceStatementSearchHit)
                        else candidate.procedure.member_block_uuids
                    )
                    if uuid in block_uuids
                ]
                passages.append(
                    SourcePassage(
                        source=matching[0].source,
                        selected_block_uuids=member_ids,
                        blocks=[
                            SourceEvidenceBlock(
                                block=SourceBlock(
                                    uuid=uuid,
                                    block_type='paragraph',
                                ),
                                page_indexes=[1],
                            )
                            for uuid in member_ids
                        ],
                        groups=[],
                    )
                )
            return passages

    repository = Repository()
    embedder = _FakeEmbedder()
    selection = _Selection([1, 0])
    service = SourceSearchService(
        repository,
        embedder,
        SourceSearchSettings(candidate_limit=16, result_limit=1),
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

    assert statement_hits == [statements[1]]
    assert procedure_hits == [procedures[1]]
    assert [candidate.is_exercise for candidate in selection.calls[0][2]] == [
        True,
        False,
    ]
    assert [item.material.text for item in selection.calls[0][2]] == [
        'exercise statement',
        'definition statement',
    ]
    assert [item.material.text for item in selection.calls[1][2]] == [
        'procedure text A',
        'procedure text B',
    ]
    assert selection.calls[0][1] is SearchMaterialKind.STATEMENT
    assert selection.calls[1][1] is SearchMaterialKind.PROCEDURE
    assert repository.calls == [
        ('statements', 'user-1', [0.4, 0.6], 16, ['source-a'], True),
        ('procedures', 'user-1', [0.4, 0.6], 16, ['source-b']),
    ]
    assert embedder.calls == [
        ([Material.from_text('exercise')], 'query'),
        ([Material.from_text('method')], 'query'),
    ]


def test_owned_occurrence_reads_preserve_source_revision_and_member_order():

    source_record = {
        'uuid': 'source-1',
        'key': 'textbook',
        'created_at': datetime(2026, 1, 1, tzinfo=UTC),
        'updated_at': datetime(2026, 1, 1, tzinfo=UTC),
        'invalidated_at': None,
        'metadata': {},
        'revision': 4,
        'identity_schema': 'source-content-v3',
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

    class Repository(SourceRetrievalRepository):
        async def read_passages(self, user_uuid, block_uuids, **kwargs):
            source = Source(uuid='source-1', key='textbook', revision=4)
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
    repository = _SourceRetrievalRepository([])
    selection = _Selection()
    service = SourceSearchService(
        repository,
        embedder,
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
    assert selection.calls == []


def test_annotation_search_preserves_governing_and_member_images(tmp_path):
    source = Source(uuid='source-1', key='book')
    instruction_image = tmp_path / 'instruction.png'
    member_image = tmp_path / 'member.png'
    Image.new('RGB', (2, 2), 'red').save(instruction_image)
    Image.new('RGB', (2, 2), 'blue').save(member_image)
    instruction_block = SourceBlock(
        uuid='instruction-block',
        block_type='image',
        content='same description',
        image_path=str(instruction_image),
        image_page_indexes=[0],
    )
    member_block = SourceBlock(
        uuid='member-block',
        block_type='image',
        content='same description',
        image_path=str(member_image),
        image_page_indexes=[1],
    )
    statement = SourceStatement(
        uuid='statement-1',
        member_block_uuids=['member-block'],
        content='statement',
        embedding=[0.1, 0.9],
    )
    procedure = SourceProcedure(
        uuid='procedure-1',
        member_block_uuids=['member-block'],
        content='procedure',
        embedding=[0.1, 0.9],
    )
    passage = SourcePassage(
        source=source,
        selected_block_uuids=['member-block'],
        blocks=[
            SourceEvidenceBlock(block=instruction_block, page_indexes=[0]),
            SourceEvidenceBlock(block=member_block, page_indexes=[1]),
        ],
        groups=[
            SourceEvidenceGroup(
                uuid='instruction-1',
                kind='instruction',
                member_block_uuids=['instruction-block'],
                governed_statement_uuids=['statement-1'],
            )
        ],
    )

    class Repository:
        async def search_owned_statements(self, *args, **kwargs):
            return [
                SourceStatementSearchHit(
                    statement=statement,
                    source=source,
                    page_indexes=[1],
                    similarity=0.9,
                )
            ]

        async def search_owned_procedures(self, *args, **kwargs):
            return [
                SourceProcedureSearchHit(
                    procedure=procedure,
                    source=source,
                    page_indexes=[1],
                    similarity=0.8,
                )
            ]

        async def read_passages(self, user_uuid, block_uuids, **kwargs):
            assert user_uuid == 'user-1'
            assert block_uuids == ['member-block']
            assert kwargs == {
                'expand_memberships': True,
                'before': 0,
                'after': 0,
            }
            return [passage]

    selection = _Selection()
    service = SourceSearchService(
        Repository(),
        _FakeEmbedder(),
        SourceSearchSettings(),
        selection=selection,
    )
    asyncio.run(service.search_statements('user-1', 'question'))
    asyncio.run(service.search_procedures('user-1', 'question'))

    assert [
        part.text
        for part in selection.calls[0][2][0].material.parts
        if isinstance(part, TextPart)
    ] == ['statement']
    assert [
        image.payload for image in selection.calls[0][2][0].material.images
    ] == [member_image, instruction_image]
    assert [
        part.text
        for part in selection.calls[1][2][0].material.parts
        if isinstance(part, TextPart)
    ] == ['procedure']
    assert [
        image.payload for image in selection.calls[1][2][0].material.images
    ] == [member_image]
