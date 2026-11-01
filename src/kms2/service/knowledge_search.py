"""User-scoped proposition embedding search and reranking."""

from kms2.config.source_search import SourceSearchSettings
from kms2.core.embedding import EmbeddingClient
from kms2.core.model.knowledge.retrieval import PropositionSearchHit
from kms2.core.reranking import RerankerClient
from kms2.database.knowledge.repository import KnowledgeRepository


class KnowledgeSearchService:
    """Search canonical propositions with shared embedding and reranking."""

    def __init__(
        self,
        repository: KnowledgeRepository,
        embedding_client: EmbeddingClient,
        reranker_client: RerankerClient,
        settings: SourceSearchSettings,
    ) -> None:
        self._repository = repository
        self._embedding_client = embedding_client
        self._reranker_client = reranker_client
        self._settings = settings

    async def search_propositions(
        self,
        user_uuid: str,
        query: str,
    ) -> list[PropositionSearchHit]:
        """Return bounded user-owned propositions in reranker order."""
        query_embedding = (await self._embedding_client.embed([query]))[0]
        candidates = await self._repository.search_propositions(
            user_uuid,
            query_embedding,
            candidate_limit=self._settings.candidate_limit,
        )
        if not candidates:
            return []
        ranked = await self._reranker_client.rerank(
            query,
            [candidate.proposition.content for candidate in candidates],
            top_n=min(self._settings.result_limit, len(candidates)),
        )
        return [
            PropositionSearchHit(
                proposition=candidates[result['index']].proposition,
                similarity=candidates[result['index']].similarity,
                relevance_score=result['relevance_score'],
            )
            for result in ranked
        ]
