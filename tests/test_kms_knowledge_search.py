"""Behavioral tests for user-scoped proposition search ranking."""

import asyncio
from datetime import UTC, datetime

from kms.config.source_search import SourceSearchSettings
from kms.core.model.knowledge.argument import ArgumentRecord
from kms.core.model.knowledge.proposition import Argument, Proposition
from kms.core.model.knowledge.retrieval import (
    ArgumentSearchCandidate,
    PropositionSearchCandidate,
)
from kms.core.model.search import SearchMaterialKind
from kms.service.knowledge_search import KnowledgeSearchService


class _Selection:
    def __init__(self, positions=None):
        self.positions = positions
        self.calls = []

    async def select(self, *, query, material_kind, candidates, content_images):
        assert content_images == []
        self.calls.append((query, material_kind, candidates))
        return (
            list(range(len(candidates)))
            if self.positions is None
            else self.positions
        )


class _Repository:
    def __init__(self, candidates: list[PropositionSearchCandidate]) -> None:
        self.candidates = candidates
        self.searched_user: str | None = None
        self.limit: int | None = None

    async def search_propositions(
        self,
        user_uuid: str,
        query_embedding: list[float],
        *,
        candidate_limit: int,
    ) -> list[PropositionSearchCandidate]:
        self.searched_user = user_uuid
        self.limit = candidate_limit
        return self.candidates


class _FakeEmbedder:
    async def embed(self, materials, *, content_images, input_type):
        assert [material.text for material in materials] == [
            'why does this property hold?'
        ]
        assert content_images == []
        assert input_type == 'query'
        return [[0.25, 0.75]]


class _Reranker:
    async def rerank(self, query, documents, *, content_images, top_n=None):
        assert query.text == 'why does this property hold?'
        assert [material.text for material in documents] == [
            'First discovered',
            'Second discovered',
        ]
        assert content_images == []
        return [
            {'index': 1, 'relevance_score': 0.96},
            {'index': 0, 'relevance_score': 0.71},
        ][:top_n]


def _proposition(uuid: str, content: str) -> Proposition:
    return Proposition(
        uuid=uuid,
        content=content,
        embedding=[0.1, 0.2],
        created_at=datetime(2025, 1, 1, tzinfo=UTC),
    )


async def _exercise_search_reranks_only_repository_owned_proposition_candidates():
    repository = _Repository(
        [
            PropositionSearchCandidate(
                proposition=_proposition('p1', 'First discovered'),
                similarity=0.83,
            ),
            PropositionSearchCandidate(
                proposition=_proposition('p2', 'Second discovered'),
                similarity=0.74,
            ),
        ]
    )
    selection = _Selection([1, 0])
    service = KnowledgeSearchService(
        repository,
        _FakeEmbedder(),
        _Reranker(),
        SourceSearchSettings(candidate_limit=12, result_limit=2),
        selection=selection,
    )

    hits = await service.search_propositions(
        'user-owned',
        'why does this property hold?',
    )

    assert repository.searched_user == 'user-owned'
    assert repository.limit == 12
    assert [hit.proposition.uuid for hit in hits] == ['p1', 'p2']
    assert [hit.similarity for hit in hits] == [0.83, 0.74]
    assert [hit.relevance_score for hit in hits] == [0.71, 0.96]
    assert selection.calls[0][1] is SearchMaterialKind.PROPOSITION
    assert [candidate.material.text for candidate in selection.calls[0][2]] == [
        'Second discovered',
        'First discovered',
    ]


def test_search_reranks_only_repository_owned_proposition_candidates():
    asyncio.run(
        _exercise_search_reranks_only_repository_owned_proposition_candidates()
    )


async def _exercise_empty_owned_candidates_do_not_call_reranker():
    class EmptyReranker:
        async def rerank(self, *args, **kwargs):
            raise AssertionError('no candidate text to rerank')

    selection = _Selection()
    service = KnowledgeSearchService(
        _Repository([]),
        _FakeEmbedder(),
        EmptyReranker(),
        SourceSearchSettings(),
        selection=selection,
    )

    assert (
        await service.search_propositions(
            'user-owned', 'why does this property hold?'
        )
        == []
    )
    assert selection.calls == []


def test_empty_owned_candidates_do_not_call_reranker():
    asyncio.run(_exercise_empty_owned_candidates_do_not_call_reranker())


def test_argument_search_selects_complete_argument_hits_without_score_changes():
    now = datetime(2025, 1, 1, tzinfo=UTC)

    def candidate(uuid: str, content: str) -> ArgumentSearchCandidate:
        return ArgumentSearchCandidate(
            record=ArgumentRecord(
                argument=Argument(
                    uuid=uuid,
                    explanation=f'{uuid} explanation',
                    embedding=[0.3, 0.7],
                    created_at=now,
                ),
                premise_uuids=['premise'],
                conclusion_uuid='conclusion',
                supporting_block_uuids=['block'],
            ),
            content=content,
            similarity=0.8 if uuid == 'argument-a' else 0.6,
        )

    candidates = [
        candidate('argument-a', 'complete argument A'),
        candidate('argument-b', 'complete argument B'),
    ]

    class Repository:
        async def search_arguments(
            self,
            user_uuid,
            query_embedding,
            *,
            candidate_limit,
        ):
            assert user_uuid == 'user-1'
            assert query_embedding == [0.25, 0.75]
            assert candidate_limit == 12
            return candidates

    class Reranker:
        async def rerank(self, query, documents, *, content_images, top_n=None):
            assert query.text == 'why does this property hold?'
            assert [material.text for material in documents] == [
                'complete argument A',
                'complete argument B',
            ]
            assert content_images == []
            return [
                {'index': 1, 'relevance_score': 0.95},
                {'index': 0, 'relevance_score': 0.61},
            ][:top_n]

    selection = _Selection([1, 0])
    service = KnowledgeSearchService(
        Repository(),
        _FakeEmbedder(),
        Reranker(),
        SourceSearchSettings(candidate_limit=12, result_limit=2),
        selection=selection,
    )

    hits = asyncio.run(
        service.search_arguments(
            'user-1',
            'why does this property hold?',
        )
    )

    assert [hit.record.argument.uuid for hit in hits] == [
        'argument-a',
        'argument-b',
    ]
    assert [hit.content for hit in hits] == [
        'complete argument A',
        'complete argument B',
    ]
    assert [hit.similarity for hit in hits] == [0.8, 0.6]
    assert [hit.relevance_score for hit in hits] == [0.61, 0.95]
    assert selection.calls[0][1] is SearchMaterialKind.ARGUMENT
    assert [candidate.material.text for candidate in selection.calls[0][2]] == [
        'complete argument B',
        'complete argument A',
    ]
