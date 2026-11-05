"""User-scoped learner-observation embedding and reranking."""

from kms.config.source_search import SourceSearchSettings
from kms.core.embedding import Embedder
from kms.core.model.content import MultimodalContent
from kms.core.model.search import (
    SearchMaterialKind,
    SearchSelectionCandidate,
)
from kms.core.model.teaching.observation import (
    LearnerObservationSearchHit,
    observation_text,
)
from kms.core.reranking import Reranker
from kms.database.teaching.observation_repository import (
    LearnerObservationRepository,
)
from kms.service.search_selection import SearchSelectionService


class LearnerObservationSearchService:
    """Retrieve and select bounded, owner-scoped observation candidates."""

    def __init__(
        self,
        repository: LearnerObservationRepository,
        embedder: Embedder,
        reranker: Reranker,
        settings: SourceSearchSettings,
        *,
        selection: SearchSelectionService,
    ) -> None:
        self._repository = repository
        self._embedder = embedder
        self._reranker = reranker
        self._settings = settings
        self._selection = selection

    async def search(
        self,
        user_uuid: str,
        query: str,
    ) -> list[LearnerObservationSearchHit]:
        """Return selected user-owned observations from the reranker shortlist."""
        query_embedding = (
            await self._embedder.embed(
                [MultimodalContent(text=query)],
                content_images=[],
                input_type='query',
            )
        )[0]
        candidates = await self._repository.search(
            user_uuid,
            query_embedding,
            candidate_limit=self._settings.candidate_limit,
        )
        if not candidates:
            return []
        ranked = await self._reranker.rerank(
            MultimodalContent(text=query),
            [
                MultimodalContent(
                    text=observation_text(
                        candidate.record.observation.kind,
                        candidate.record.observation.topic,
                        candidate.record.observation.description,
                    )
                )
                for candidate in candidates
            ],
            content_images=[],
            top_n=min(self._settings.result_limit, len(candidates)),
        )
        hits = [
            LearnerObservationSearchHit(
                record=candidates[result['index']].record,
                similarity=candidates[result['index']].similarity,
                relevance_score=result['relevance_score'],
            )
            for result in ranked
        ]
        if not hits:
            return []
        positions = await self._selection.select(
            query=query,
            material_kind=SearchMaterialKind.LEARNER_OBSERVATION,
            candidates=[
                SearchSelectionCandidate(
                    position=position,
                    material=MultimodalContent(
                        text=observation_text(
                            hit.record.observation.kind,
                            hit.record.observation.topic,
                            hit.record.observation.description,
                        )
                    ),
                )
                for position, hit in enumerate(hits)
            ],
            content_images=[],
        )
        return [hits[position] for position in positions]
