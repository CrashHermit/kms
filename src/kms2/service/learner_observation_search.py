"""User-scoped learner-observation embedding and reranking."""

from kms2.config.source_search import SourceSearchSettings
from kms2.core.embedding import EmbeddingClient
from kms2.core.model.teaching.observation import (
    LearnerObservationSearchHit,
    observation_text,
)
from kms2.core.reranking import RerankerClient
from kms2.database.teaching.observation_repository import (
    LearnerObservationRepository,
)


class LearnerObservationSearchService:
    """Retrieve bounded owner-scoped observation candidates and rerank them."""

    def __init__(
        self,
        repository: LearnerObservationRepository,
        embedding_client: EmbeddingClient,
        reranker_client: RerankerClient,
        settings: SourceSearchSettings,
    ) -> None:
        self._repository = repository
        self._embedding_client = embedding_client
        self._reranker_client = reranker_client
        self._settings = settings

    async def search(
        self,
        user_uuid: str,
        query: str,
    ) -> list[LearnerObservationSearchHit]:
        """Return retrieved observations in actual reranker order."""
        query_embedding = (await self._embedding_client.embed([query]))[0]
        candidates = await self._repository.search(
            user_uuid,
            query_embedding,
            candidate_limit=self._settings.candidate_limit,
        )
        if not candidates:
            return []
        ranked = await self._reranker_client.rerank(
            query,
            [
                observation_text(
                    candidate.record.observation.kind,
                    candidate.record.observation.topic,
                    candidate.record.observation.description,
                )
                for candidate in candidates
            ],
            top_n=min(self._settings.result_limit, len(candidates)),
        )
        return [
            LearnerObservationSearchHit(
                record=candidates[result['index']].record,
                similarity=candidates[result['index']].similarity,
                relevance_score=result['relevance_score'],
            )
            for result in ranked
        ]
