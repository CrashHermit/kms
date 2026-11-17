"""Behavioral tests for owner-scoped source block and unit search."""

import asyncio
from pathlib import Path

from kms.core.embedding import EmbeddingRequest
from kms.media.models import ImageAsset, ImagePart, Material, TextPart
from kms.retrieval.selection.models import SearchMaterialKind
from kms.source.models.source import Source, SourceBlock
from kms.source.models.unit import SourceUnit, SourceUnitKind
from kms.source.retrieval.models import (
    SourceBlockRead,
    SourceSearchHit,
    SourceUnitRead,
    SourceUnitSearchHit,
)
from kms.source.retrieval.search import SourceSearchService
from kms.source.retrieval.settings import SourceSearchSettings


class _FakeEmbedder:
    def __init__(self) -> None:
        self.calls = []

    async def embed(self, request: EmbeddingRequest):
        self.calls.append((request.materials, request.input_type))
        return [[0.4, 0.6]]


class _Repository:
    def __init__(self, blocks=None, units=None) -> None:
        self.blocks = blocks or []
        self.units = units or []
        self.calls = []

    async def search_owned_blocks(
        self, user_uuid, query_embedding, *, candidate_limit, source_uuids=None
    ):
        self.calls.append(
            (
                'blocks',
                user_uuid,
                query_embedding,
                candidate_limit,
                source_uuids,
            )
        )
        return self.blocks

    async def search_owned_units(
        self,
        user_uuid,
        query_embedding,
        *,
        candidate_limit,
        source_uuids=None,
        kind=None,
    ):
        self.calls.append(
            (
                'units',
                user_uuid,
                query_embedding,
                candidate_limit,
                source_uuids,
                kind,
            )
        )
        return self.units


class _Selection:
    def __init__(self, positions=None) -> None:
        self.positions = positions
        self.calls = []

    async def select(self, *, query, material_kind, candidates):
        self.calls.append((query, material_kind, candidates))
        return (
            list(range(len(candidates)))
            if self.positions is None
            else self.positions
        )


def _block_hit(block_uuid: str, *, content: str, similarity: float):
    return SourceSearchHit(
        source=Source(uuid=f'{block_uuid}-source', key='book.pdf'),
        unit_uuid=f'{block_uuid}-unit',
        block=SourceBlock(
            uuid=block_uuid,
            block_type='paragraph',
            content=content,
        ),
        similarity=similarity,
    )


def _unit_hit(
    unit_uuid: str, kind: SourceUnitKind, content: str, similarity: float
):
    return SourceUnitSearchHit(
        source=Source(uuid=f'{unit_uuid}-source', key='book.pdf'),
        unit=SourceUnit(
            uuid=unit_uuid,
            kind=kind,
            blocks=[
                SourceBlock(
                    uuid=f'{unit_uuid}-block',
                    block_type='paragraph',
                    content=content,
                ),
            ],
            embedding=[0.1, 0.9],
        ),
        similarity=similarity,
    )


def test_block_search_preserves_candidate_order_and_result_limit():
    candidates = [
        _block_hit('block-a', content='Candidate A', similarity=0.92),
        _block_hit('block-b', content='Candidate B', similarity=0.88),
        _block_hit('block-c', content='Candidate C', similarity=0.81),
    ]
    embedder = _FakeEmbedder()
    repository = _Repository(blocks=candidates)
    selection = _Selection([2, 0, 1])
    service = SourceSearchService(
        repository,
        embedder,
        SourceSearchSettings(candidate_limit=12, result_limit=2),
        selection=selection,
    )

    hits = asyncio.run(
        service.search_sources(
            'user-1', Material.from_text('query'), source_uuids=['source-2']
        )
    )
    assert hits == [candidates[2], candidates[0]]
    assert [(hit.block.uuid, hit.unit_uuid) for hit in hits] == [
        ('block-c', 'block-c-unit'),
        ('block-a', 'block-a-unit'),
    ]

    assert embedder.calls == [([Material.from_text('query')], 'query')]
    assert repository.calls == [
        ('blocks', 'user-1', [0.4, 0.6], 12, ['source-2'])
    ]
    assert selection.calls[0][1] is SearchMaterialKind.SOURCE_BLOCK
    assert [item.material.text for item in selection.calls[0][2]] == [
        'Candidate A',
        'Candidate B',
        'Candidate C',
    ]
    assert [item.position for item in selection.calls[0][2]] == [0, 1, 2]


def test_block_search_preserves_image_and_interleaved_query_material():
    query_image = ImageAsset(
        key=('path', '/queries/figure.png'),
        payload=Path('/queries/figure.png'),
    )
    queries = [
        Material(parts=(ImagePart(asset=query_image),)),
        Material(
            parts=(
                TextPart(text='compare'),
                ImagePart(asset=query_image),
                TextPart(text='with this'),
            )
        ),
    ]
    candidates = [
        SourceSearchHit(
            source=Source(uuid='source-1', key='book.pdf'),
            unit_uuid='unit-1',
            block=SourceBlock(
                uuid='image-only',
                block_type='image',
                image_path='/assets/figure.png',
            ),
            similarity=0.8,
        ),
        SourceSearchHit(
            source=Source(uuid='source-1', key='book.pdf'),
            unit_uuid='unit-1',
            block=SourceBlock(
                uuid='mixed',
                block_type='image',
                content='figure explanation',
                image_path='/assets/figure.png',
            ),
            similarity=0.7,
        ),
    ]
    embedder = _FakeEmbedder()
    selection = _Selection()
    service = SourceSearchService(
        _Repository(blocks=candidates),
        embedder,
        SourceSearchSettings(),
        selection=selection,
    )

    for query in queries:
        assert (
            asyncio.run(service.search_sources('user-1', query)) == candidates
        )

    assert [call[0] for call in embedder.calls] == [
        [queries[0]],
        [queries[1]],
    ]
    assert all(
        embedder.calls[index][0][0] is query
        for index, query in enumerate(queries)
    )
    assert [call[1] for call in embedder.calls] == ['query', 'query']
    assert all(
        selection.calls[index][0] is query
        for index, query in enumerate(queries)
    )
    assert [part for part in selection.calls[0][2][0].material.parts] == [
        ImagePart(
            asset=ImageAsset(
                key=('path', '/assets/figure.png'),
                payload=Path('/assets/figure.png'),
            )
        )
    ]
    assert [type(part) for part in selection.calls[0][2][1].material.parts] == [
        TextPart,
        ImagePart,
    ]
    assert [
        part.text
        for part in selection.calls[0][2][1].material.parts
        if isinstance(part, TextPart)
    ] == ['figure explanation']


def test_unit_search_preserves_image_and_interleaved_query_material():
    query_image = ImageAsset(
        key=('path', '/queries/figure.png'),
        payload=Path('/queries/figure.png'),
    )
    queries = [
        Material(parts=(ImagePart(asset=query_image),)),
        Material(
            parts=(
                TextPart(text='compare'),
                ImagePart(asset=query_image),
                TextPart(text='with this'),
            )
        ),
    ]
    first = _unit_hit('unit-a', SourceUnitKind.EXERCISE, 'exercise', 0.7)
    first.unit.blocks = [
        SourceBlock(
            uuid='text-before',
            block_type='paragraph',
            content='before',
        ),
        SourceBlock(
            uuid='image',
            block_type='image',
            image_path='/assets/figure.png',
        ),
        SourceBlock(
            uuid='text-after',
            block_type='paragraph',
            content='after',
        ),
    ]
    image_only = _unit_hit('unit-image', SourceUnitKind.EXERCISE, '', 0.6)
    image_only.unit.blocks = [
        SourceBlock(
            uuid='unit-image-block',
            block_type='image',
            image_path='/assets/unit-image.png',
        )
    ]
    repository = _Repository(units=[first, image_only])
    selection = _Selection([0])
    embedder = _FakeEmbedder()
    service = SourceSearchService(
        repository,
        embedder,
        SourceSearchSettings(candidate_limit=16, result_limit=1),
        selection=selection,
    )

    for query in queries:
        assert asyncio.run(
            service.search_source_units(
                'user-1',
                query,
                source_uuids=['source-a'],
                kind=SourceUnitKind.EXERCISE,
            )
        ) == [first]

    assert all(
        embedder.calls[index][0][0] is query
        for index, query in enumerate(queries)
    )
    assert [call[1] for call in embedder.calls] == ['query', 'query']
    assert all(
        selection.calls[index][0] is query
        for index, query in enumerate(queries)
    )
    assert all(
        selection.calls[index][1] is SearchMaterialKind.SOURCE_UNIT
        for index in range(len(queries))
    )
    selected_candidates = selection.calls[0][2]
    assert selected_candidates[0].unit_kind == SourceUnitKind.EXERCISE.value
    assert [type(part) for part in selected_candidates[0].material.parts] == [
        TextPart,
        ImagePart,
        TextPart,
    ]
    assert selected_candidates[0].material.parts[1].asset == ImageAsset(
        key=('path', '/assets/figure.png'),
        payload=Path('/assets/figure.png'),
    )
    assert [
        part.text
        for part in selected_candidates[0].material.parts
        if isinstance(part, TextPart)
    ] == ['before', 'after']
    assert [type(part) for part in selected_candidates[1].material.parts] == [
        ImagePart
    ]
    assert selected_candidates[1].material.images == (
        ImageAsset(
            key=('path', '/assets/unit-image.png'),
            payload=Path('/assets/unit-image.png'),
        ),
    )
    assert (
        repository.calls
        == [
            (
                'units',
                'user-1',
                [0.4, 0.6],
                16,
                ['source-a'],
                SourceUnitKind.EXERCISE,
            )
        ]
        * 2
    )
    assert [call[0] for call in embedder.calls] == [
        [queries[0]],
        [queries[1]],
    ]


def test_empty_search_results_and_allowlists_return_no_hits():
    for search in ('blocks', 'units'):
        embedder = _FakeEmbedder()
        repository = _Repository()
        selection = _Selection()
        service = SourceSearchService(
            repository,
            embedder,
            SourceSearchSettings(),
            selection=selection,
        )
        if search == 'blocks':
            query = Material.from_text('query')
            result = asyncio.run(service.search_sources('user-1', query))
            empty = asyncio.run(
                service.search_sources('user-1', query, source_uuids=[])
            )
        else:
            query = Material.from_text('query')
            result = asyncio.run(service.search_source_units('user-1', query))
            empty = asyncio.run(
                service.search_source_units('user-1', query, source_uuids=[])
            )
        assert result == empty == []
        assert len(embedder.calls) == 1
        assert len(repository.calls) == 1


def test_retrieval_read_material_properties_are_read_time_projections():
    source = Source(uuid='source-1', key='book.pdf')
    block = SourceBlock(
        uuid='block-1',
        block_type='paragraph',
        content='block text',
    )
    unit = SourceUnit(
        uuid='unit-1',
        kind=SourceUnitKind.EXERCISE,
        blocks=[block],
    )

    block_read = SourceBlockRead(
        source=source, unit_uuid=unit.uuid, block=block
    )
    unit_read = SourceUnitRead(source=source, unit=unit)

    assert block_read.material == Material.from_text('block text')
    assert unit_read.material == Material.from_text('block text')
    assert 'material' not in block_read.model_dump()
    assert 'material' not in unit_read.model_dump()
