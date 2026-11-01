"""Behavioral tests for user-scoped proposition search ranking."""

import asyncio
from datetime import UTC, datetime

from kms2.config.source_search import SourceSearchSettings
from kms2.core.model.knowledge.proposition import Proposition
from kms2.core.model.knowledge.retrieval import PropositionSearchCandidate
from kms2.service.knowledge_search import KnowledgeSearchService


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


class _EmbeddingClient:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        assert texts == ['why does this property hold?']
        return [[0.25, 0.75]]


class _Reranker:
    async def rerank(self, query, documents, top_n=None):
        assert documents == ['First discovered', 'Second discovered']
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
    service = KnowledgeSearchService(
        repository,
        _EmbeddingClient(),
        _Reranker(),
        SourceSearchSettings(candidate_limit=12, result_limit=2),
    )

    hits = await service.search_propositions(
        'user-owned',
        'why does this property hold?',
    )

    assert repository.searched_user == 'user-owned'
    assert repository.limit == 12
    assert [hit.proposition.uuid for hit in hits] == ['p2', 'p1']
    assert [hit.similarity for hit in hits] == [0.74, 0.83]
    assert [hit.relevance_score for hit in hits] == [0.96, 0.71]


def test_search_reranks_only_repository_owned_proposition_candidates():
    asyncio.run(
        _exercise_search_reranks_only_repository_owned_proposition_candidates()
    )


async def _exercise_empty_owned_candidates_do_not_call_reranker():
    class EmptyReranker:
        async def rerank(self, *args, **kwargs):
            raise AssertionError('no candidate text to rerank')

    service = KnowledgeSearchService(
        _Repository([]),
        _EmbeddingClient(),
        EmptyReranker(),
        SourceSearchSettings(),
    )

    assert (
        await service.search_propositions(
            'user-owned', 'why does this property hold?'
        )
        == []
    )


def test_empty_owned_candidates_do_not_call_reranker():
    asyncio.run(_exercise_empty_owned_candidates_do_not_call_reranker())
